import time
import io
import pdfplumber


class RAGError(Exception):
    """Raised when the RAG pipeline encounters a recoverable error."""
    pass


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract plain text from a PDF file uploaded via Streamlit.

    Args:
        uploaded_file: A Streamlit UploadedFile object (BytesIO-compatible).

    Returns:
        Extracted text as a single string, with pages separated by newlines.

    Raises:
        ValueError: If the PDF is empty or text extraction fails.
    """
    try:
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError(
                "The PDF appears to be empty or contains only images (no extractable text)."
            )
        return text
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")


def validate_text(text: str, name: str = "Input", min_length: int = 10) -> str:
    """Validate that a text input is non-empty and meets a minimum length.

    Raises:
        ValueError: If the text is empty or too short.
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"{name} cannot be empty.")
    stripped = text.strip()
    if len(stripped) < min_length:
        raise ValueError(
            f"{name} is too short (minimum {min_length} characters). Got: {len(stripped)}"
        )
    return stripped


def safe_answer(rag, query: str, preprocess_fn, top_k: int = 5, max_retries: int = 2) -> str:
    """Call rag.answer() with retry logic and graceful error handling.

    Returns:
        The generated answer string, or a human-readable error message.
    """
    for attempt in range(max_retries):
        try:
            validate_text(query, name="Query")
            return rag.answer(query, preprocess_fn, top_k)
        except RAGError as e:
            return f"RAG Error: {str(e)}"
        except ValueError as e:
            return f"Validation Error: {str(e)}"
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "Rate limit reached. Please try again in a moment."
            if "Connection refused" in err_str or "Failed to connect" in err_str:
                return (
                    "⚠️ Could not connect to Ollama. "
                    "Make sure Ollama is running (`ollama serve`) "
                    "and the model is pulled (`ollama pull llama3.1`)."
                )
            return f"Something went wrong: {err_str}"
    return "Max retries exceeded. Please try again later."
