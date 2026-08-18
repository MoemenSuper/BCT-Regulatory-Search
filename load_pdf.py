from docling_pdf_loader import DoclingPdfLoader


_loader = DoclingPdfLoader()


def load_pdf(file_path: str, *, force_ocr: bool = False):
    return _loader.load(file_path, force_ocr=force_ocr)

