import zipfile
import os
import argparse
from pathlib import Path


def unzip(input_path, output_path):
    """
    Unzip a file from input_path to output_path.
    
    Args:
        input_path (str or Path): Path to the zip file to extract
        output_path (str or Path): Path to the directory where files will be extracted
    
    Returns:
        str: Path to the extracted directory
    
    Raises:
        FileNotFoundError: If the input zip file doesn't exist
        zipfile.BadZipFile: If the input file is not a valid zip file
        PermissionError: If there are permission issues during extraction
    """
    # Convert to Path objects for easier handling
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Zip file not found: {input_path}")
    
    # Check if input file is a zip file
    if not input_path.suffix.lower() == '.zip':
        raise ValueError(f"Input file is not a zip file: {input_path}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            # Extract all files to the output directory
            zip_ref.extractall(output_path)
            
            # Get list of extracted files for verification
            extracted_files = zip_ref.namelist()
            
        print(f"Successfully extracted {len(extracted_files)} files from {input_path} to {output_path}")
        return str(output_path)
        
    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Invalid zip file: {input_path}")
    except PermissionError:
        raise PermissionError(f"Permission denied when extracting to: {output_path}")
    except Exception as e:
        raise Exception(f"Unexpected error during extraction: {str(e)}")


def unzip_with_progress(input_path, output_path, show_progress=True):
    """
    Unzip a file with optional progress display.
    
    Args:
        input_path (str or Path): Path to the zip file to extract
        output_path (str or Path): Path to the directory where files will be extracted
        show_progress (bool): Whether to show extraction progress
    
    Returns:
        str: Path to the extracted directory
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Zip file not found: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            if show_progress:
                print(f"Extracting {total_files} files...")
            
            for i, file_name in enumerate(file_list, 1):
                zip_ref.extract(file_name, output_path)
                
                if show_progress and i % max(1, total_files // 10) == 0:
                    progress = (i / total_files) * 100
                    print(f"Progress: {progress:.1f}% ({i}/{total_files} files)")
            
            if show_progress:
                print(f"Extraction complete! All files extracted to {output_path}")
                
        return str(output_path)
        
    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Invalid zip file: {input_path}")
    except Exception as e:
        raise Exception(f"Error during extraction: {str(e)}")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Unzip a file to a specified directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unzip.py data.zip extracted_data/
  python unzip.py --input data.zip --output Dataset/
  python unzip.py -i large_file.zip -o results/ --progress
        """
    )
    
    parser.add_argument(
        'input_path', 
        nargs='?',
        help='Path to the zip file to extract'
    )
    
    parser.add_argument(
        'output_path', 
        nargs='?',
        help='Path to the directory where files will be extracted'
    )
    
    parser.add_argument(
        '-i', '--input',
        dest='input_path_flag',
        help='Path to the zip file to extract (alternative to positional argument)'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_path_flag',
        help='Path to the directory where files will be extracted (alternative to positional argument)'
    )
    
    parser.add_argument(
        '-p', '--progress',
        action='store_true',
        help='Show extraction progress'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to handle command line execution.
    """
    args = parse_arguments()
    
    # Determine input and output paths
    input_path = args.input_path or args.input_path_flag
    output_path = args.output_path or args.output_path_flag
    
    # Validate that both paths are provided
    if not input_path:
        print("Error: Input path is required")
        print("Use: python unzip.py <input_path> <output_path>")
        print("Or: python unzip.py --input <input_path> --output <output_path>")
        return 1
    
    if not output_path:
        print("Error: Output path is required")
        print("Use: python unzip.py <input_path> <output_path>")
        print("Or: python unzip.py --input <input_path> --output <output_path>")
        return 1
    
    try:
        if args.progress:
            unzip_with_progress(input_path, output_path, show_progress=True)
        else:
            unzip(input_path, output_path)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


# Example usage
if __name__ == "__main__":
    exit(main())
