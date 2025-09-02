import asyncio

# Constants for streaming
MIN_BUFFER_LENGTH = 50
MAX_BUFFER_LENGTH = 1000
SENTENCE_END_CHARS = {".", "!", "?"}
YIELD_CHARS = SENTENCE_END_CHARS.union({"\n", " "})


class StreamManager:
    """Manages active streaming connections."""

    def __init__(self):
        self.active_streams = set()
        self.lock = asyncio.Lock()

    async def add_stream(self, stream_id: str):
        async with self.lock:
            self.active_streams.add(stream_id)

    async def remove_stream(self, stream_id: str):
        async with self.lock:
            self.active_streams.discard(stream_id)

    async def is_stream_active(self, stream_id: str) -> bool:
        async with self.lock:
            return stream_id in self.active_streams


async def process_chunk(content: str, response_buffer: str) -> tuple[str, bool]:
    """
    Processes a chunk of content from the LLM stream and determines if
    a yieldable sentence or fragment has been formed.
    """
    if not content:
        return response_buffer, False

    response_buffer += content
    trimmed_buffer = response_buffer.rstrip()

    if not trimmed_buffer:
        return response_buffer, False

    last_char = trimmed_buffer[-1]
    buffer_len = len(response_buffer)

    # Determine if the buffer should be yielded
    should_yield = (
        (buffer_len > 5 and (last_char in SENTENCE_END_CHARS or last_char == "\n"))
        or (buffer_len > MIN_BUFFER_LENGTH and last_char == " ")
        or (buffer_len > MAX_BUFFER_LENGTH)
    )

    return response_buffer, should_yield
