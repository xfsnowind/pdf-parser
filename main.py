from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from vision_parse import VisionParser
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import img2pdf
from PIL import Image

app = FastAPI(title="PDF Parser API")


def save_markdown_pages(markdown_pages: list[str], output_path: str) -> str:
    """Save markdown pages to a file and return the file path."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Combine all pages with page separators
    content = ""
    for i, page_content in enumerate(markdown_pages):
        content += f"\n## Page {i+1}\n\n{page_content}\n\n---\n"

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


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
        f.write(img2pdf.convert(image_path))

    # Clean up RGB temporary file if it was created
    if image_path.endswith("_rgb.jpg"):
        os.unlink(image_path)

    return pdf_path


@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
        return {"error": "File must be a PDF or image (jpg, jpeg, png)"}

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
            max_output_tokens=2048,
            image_mode="url",
            detailed_extraction=True,
        )

        # Convert the file
        markdown_pages = parser.convert_pdf(file_to_parse)

        # Generate output filename based on input filename
        output_filename = os.path.splitext(file.filename)[0] + ".md"
        output_path = f"./output/{output_filename}"

        # Save markdown pages to file
        saved_path = save_markdown_pages(markdown_pages, output_path)

        return {
            "message": "File successfully converted to markdown",
            "saved_path": saved_path,
            "total_pages": len(markdown_pages),
            "original_filename": file.filename,
            "file_type": "image" if is_image else "pdf",
        }

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
