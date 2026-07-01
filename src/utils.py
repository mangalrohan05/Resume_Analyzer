import time
import io
import pdfplumber


class RAGError(Exception):
    pass


def extract_text_from_pdf(uploaded_file) -> str:

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

    if not text or not isinstance(text, str):
        raise ValueError(f"{name} cannot be empty.")
    stripped = text.strip()
    if len(stripped) < min_length:
        raise ValueError(
            f"{name} is too short (minimum {min_length} characters). Got: {len(stripped)}"
        )
    return stripped


def safe_answer(rag, query: str, preprocess_fn, top_k: int = 5, max_retries: int = 2) -> str:
    """Call rag.answer() with retry logic and graceful Gemini error handling."""
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
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "⚠️ Gemini rate limit reached. Please wait a moment and try again."
            if "API_KEY_INVALID" in err_str or "invalid api key" in err_str.lower():
                return (
                    "⚠️ Invalid Gemini API key. "
                    "Check that GEMINI_API_KEY is set correctly in your .env file."
                )
            if "GEMINI_API_KEY" in err_str:
                return (
                    "⚠️ GEMINI_API_KEY not found. "
                    "Add it to your .env file: GEMINI_API_KEY=your_key_here"
                )
            return f"Something went wrong: {err_str}"
    return "Max retries exceeded. Please try again later."
