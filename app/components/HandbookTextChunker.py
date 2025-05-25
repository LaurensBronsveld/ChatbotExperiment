import re


from app.components.TextChunker import TextChunker


class HandbookTextChunker(TextChunker):
    HANDBOOK_ROOT_URL = (
        "https://gitlab.com/gitlab-com/content-sites/handbook/-/tree/main/content"
    )

    def create_metadata(self, headers: str):
        """
        Creates metadata from a string of headers separated by newlines.

        Args:
            headers: A string containing headers separated by newline characters ('\n').

        Returns:
            A dictionary containing the headers as a list under the key "headers".
            Returns an empty dictionary if the input string is empty.
        """

        if not headers:
            return {"headers": []}  # Return an empty list if headers string is empty

        headers_list = headers.split("\n")
        # Remove any empty strings that might result from trailing newlines
        headers_list = [header.strip() for header in headers_list if header]
        metadata = {"headers": headers_list}
        return metadata

    def split_text_into_chunks(self):
        md_regex = (
            r"(^#+\s*.*)"  # regex which captures all levels of headers in markdown.
        )
        chunks = []

        # Helper function to add a merged chunk if present.
        def flush_temp_chunk():
            nonlocal temp_chunk, temp_headers
            if temp_chunk:
                self.add_chunk(doc_id, temp_chunk, self.create_metadata(temp_headers))
                temp_chunk, temp_headers = "", ""

        for doc in self.docs:
            temp_chunk = ""
            temp_headers = ""
            doc_id = doc["doc_id"]
            text = doc["text"]
            # split text by headers
            sections = re.split(md_regex, text, flags=re.MULTILINE)

            # capture first text which often does not start with a header
            if len(self.tokenizer.encode(sections[0])) < self.min_tokens:
                temp_chunk += sections[0] + "\n"
            else:
                self.add_chunk(doc_id, sections[0], self.create_metadata(header))

            for i in range(
                1, len(sections), 2
            ):  # loop through headers and text in sections
                header = sections[i].strip()
                content = sections[i + 1].strip() if i + 1 <= len(sections) else ""

                token_count = len(self.tokenizer.encode(content))

                # add chunk to chunk list or to temporary chunk to combine with other chunks
                if token_count < self.min_tokens:
                    temp_chunk += content + "\n"
                    temp_headers += header + "\n"
                else:
                    # add temp chunk if it exists
                    flush_temp_chunk()

                    self.add_chunk(doc_id, text, self.create_metadata(header))

            # add remaining temp chunk if it exists
            flush_temp_chunk()

        return chunks
