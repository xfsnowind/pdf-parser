from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from vision_parse import VisionParser
import os
from tempfile import NamedTemporaryFile
import img2pdf
from PIL import Image
import tempfile


app = FastAPI(title="PDF Parser API")


def format_markdown_content(markdown_pages: list[str]) -> str:
    """Format markdown pages into a single string with page separators."""
    content = ""
    for i, page_content in enumerate(markdown_pages):
        content += f"\n## Page {i+1}\n\n{page_content}\n\n---\n"
    return content


def convert_image_to_pdf(image_path: str) -> str:
    """Convert image to PDF and return the path to the PDF file."""
    # Open image to check format and convert if necessary
    with Image.open(image_path) as img:
        # If image is not in RGB mode, convert it
        if img.mode != "RGB":
            img = img.convert("RGB")
            # Save as temporary RGB image
            rgb_image_path = image_path + "_rgb.jpg"
            img.save(rgb_image_path, "JPEG")
            image_path = rgb_image_path

    # Convert to PDF
    pdf_path = image_path + ".pdf"
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_path, rotation=img2pdf.Rotation.ifvalid))

    # Clean up RGB temporary file if it was created
    if image_path.endswith("_rgb.jpg"):
        os.unlink(image_path)

    return pdf_path


@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
        return JSONResponse(
            status_code=400,
            content={"error": "File must be a PDF or image (jpg, jpeg, png)"},
        )

    # Create a temporary file to store the uploaded content
    with NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Convert image to PDF if necessary
        file_to_parse = temp_file_path
        is_image = file.filename.lower().endswith((".jpg", ".jpeg", ".png"))

        if is_image:
            file_to_parse = convert_image_to_pdf(temp_file_path)

        # Initialize parser
        parser = VisionParser(
            model_name="gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.9,
            top_p=0.4,
            max_output_tokens=8192,
            image_mode="url",
            detailed_extraction=True,
        )

        # Convert the file
        markdown_pages = parser.convert_pdf(file_to_parse)

        # Format markdown content
        markdown_content = format_markdown_content(markdown_pages)

        # Create a temporary markdown file
        output_filename = os.path.splitext(file.filename)[0] + ".md"
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".md"
        ) as md_file:
            md_file.write(markdown_content)
            md_file_path = md_file.name

        # Return the file as a downloadable response
        return FileResponse(
            md_file_path,
            media_type="text/markdown",
            filename=output_filename,
            background=None,  # This ensures the file is deleted after sending
        )

    finally:
        # Clean up temporary files
        os.unlink(temp_file_path)
        if is_image and file_to_parse != temp_file_path:
            os.unlink(file_to_parse)


@app.get("/")
async def root():
    return {
        "message": "Welcome to PDF Parser API",
        "endpoints": {
            "POST /parse": "Upload a PDF or image file to convert it to markdown"
        },
    }
