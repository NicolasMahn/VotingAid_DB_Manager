import os
import json
import PyPDF2


def find_pages_for_paragraphs(pdf_path, paragraphs, start_page=1):
    """
    Finds pages for multiple paragraphs in a PDF, starting from a given page.

    Args:
        pdf_path (str): Path to the PDF file.
        paragraphs (list of dict): A list of dictionaries with 'heading' and 'content'.
        start_page (int): The page number to start searching from (1-based index).

    Returns:
        list of dict: A list of dictionaries with 'heading', 'content', and 'page'.
    """
    results = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

            for paragraph in paragraphs:
                heading = paragraph["heading"]
                content = paragraph["content"]
                found_page = -1

                # Search for the page of the heading, starting from the specified page
                for page_number in range(start_page - 1, num_pages):
                    text = reader.pages[page_number].extract_text()
                    if heading in text:
                        found_page = page_number + 1  # Convert 0-based index to 1-based
                        break

                # Store result
                results.append({
                    "heading": heading,
                    "content": content,
                    "page": found_page
                })
    except Exception as e:
        print(f"Error reading the PDF: {e}")

    return results


def save_results(output_dir, pdf_name, results):
    """
    Saves the extracted sections as text files and updates the JSON configuration file.

    Args:
        output_dir (str): Base directory for saving output.
        pdf_name (str): Original PDF file name.
        results (list of dict): List of results with 'heading', 'content', and 'page'.
    """
    documents_dir = os.path.join(output_dir, "documents")
    config_path = os.path.join(output_dir, "data_config.json")

    # Ensure the output directory exists
    os.makedirs(documents_dir, exist_ok=True)

    # Load or initialize the configuration file
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = []

    # Process each result
    for result in results:
        heading = result["heading"]
        content = result["content"]
        page = result["page"]
        txt_filename = f"{heading.replace(' ', '_').replace(':', '').replace('/', '')}.txt"
        txt_path = os.path.join(documents_dir, txt_filename)

        # Save the content as a text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Add entry to the configuration
        config_entry = {
            "pdf_name": pdf_name,
            "txt_file": txt_filename,
            "page": page,
            "title": heading
        }
        config.append(config_entry)

    # Save the updated configuration file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def multiline_input(prompt):
    """
    Reads multiline input until the user types 'END'.

    Args:
        prompt (str): The prompt message for the input.

    Returns:
        str: The complete multiline input.
    """
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    # Input paths
    pdf_path = input("Enter the path to the PDF file: ").strip()
    output_dir = input("Enter the output directory: ").strip()

    # Specify start page for searching
    start_page = int(input("Enter the starting page for search (1-based index): ").strip())

    # Input paragraphs
    paragraphs = []
    print("Enter the headings and content. Type 'STOP' to finish or 'BACK' to reenter the last entry.")
    last_entry = None
    while True:
        heading = input("Enter heading (or 'STOP' to finish, 'BACK' to reenter): ").strip()
        if heading.lower() == "stop":
            break
        if heading.lower() == "back":
            if last_entry:
                print("Reentering the last entry:")
                print(f"Previous heading: {last_entry['heading']}")
                print(f"Previous content:\n{last_entry['content']}")
                heading = input("Enter new heading (or press Enter to keep the same): ").strip()
                if heading == "":
                    heading = last_entry["heading"]
                content = multiline_input("Enter new content (or type 'END' to finish): ")
                if content.strip() == "":
                    content = last_entry["content"]
                paragraphs[-1] = {"heading": heading, "content": content}
            else:
                print("No previous entry to reenter.")
            continue

        content = multiline_input("Enter content (type 'END' to finish): ")
        last_entry = {"heading": heading, "content": content}
        paragraphs.append(last_entry)

    # Find pages for paragraphs starting from the specified page
    pdf_name = os.path.basename(pdf_path)
    results = find_pages_for_paragraphs(pdf_path, paragraphs, start_page)

    # Save results
    save_results(output_dir, pdf_name, results)
    print("Results saved successfully.")


if __name__ == "__main__":
    main()
