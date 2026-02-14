import sys
import os

# Add parent directory to path to allow importing lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.custom_types import MatrixLanguage, EmbeddedLanguage
import argostranslate.package
import argostranslate.translate

def install_languages():
    print("Updating package index...")
    argostranslate.package.update_package_index()
    
    available_packages = argostranslate.package.get_available_packages()
    
    # We need pairs involving English to pivot
    # Dynamic generation from custom_types
    matrix_langs = {v for k, v in MatrixLanguage.__dict__.items() if not k.startswith('__')}
    embedded_langs = {v for k, v in EmbeddedLanguage.__dict__.items() if not k.startswith('__')}
    
    all_langs = matrix_langs.union(embedded_langs)
    
    # Remove 'en' from the set because we are pairing *with* English
    if 'en' in all_langs:
        all_langs.remove('en')
        
    required_codes = all_langs
    
    print(f"Looking for models connecting English (en) with: {required_codes}")
    
    count = 0
    for package in available_packages:
        # Check if it's an English pair
        if package.from_code == 'en' and package.to_code in required_codes:
            print(f"Installing {package.from_code} -> {package.to_code}...")
            argostranslate.package.install_from_path(package.download())
            count += 1
        elif package.to_code == 'en' and package.from_code in required_codes:
            print(f"Installing {package.from_code} -> {package.to_code}...")
            argostranslate.package.install_from_path(package.download())
            count += 1
            
    print(f"Installed {count} models.")

if __name__ == "__main__":
    install_languages()
