import re
def check_template(text):
    ###
    # This function helps determine whether the user inputs the correct format for the specialized question (for disease diagnosis)
    # Age:
    # Sex:
    # Ethnicity:
    # Race:
    # Phenotypes:
    # ~~~
    # Candidate Genes from genome/exome sequencing tests:
    # ~~~
    # What is the likely diagnosis?
    # ###
    patterns = {
        "age": r"Age:\s+\d+\s+(months|years)\s+old",
        "sex": r"Sex:\s+(Male|Female)",
        "ethnicity": r"Ethnicity:\s+[A-Za-z]+",
        "race": r"Race:\s+[A-Za-z]+",
        "phenotypes": r"Phenotypes:\s+([A-Za-z]+(,\s*)?)+",
        "genes": r"Candidate Genes from genome/exome sequencing tests:\s+([A-Za-z0-9])",
        "question": r"\s+What+\s+[A-Za-z]+"
    }

    # Split text into meaningful sections
    lines = [line.strip() for line in text.strip().split("|") if line.strip()]
    # Check the fixed sections
    try:
        # Validate each pattern in sequence
        if not re.match(patterns["age"], lines[0]):
            print(0)
            return False
        if not re.match(patterns["sex"], lines[1]):
            print(1)
            return False
        if not re.match(patterns["ethnicity"], lines[2]):
            print(2)
            return False
        if not re.match(patterns["race"], lines[3]):
            print(3)
            return False
        if not re.match(patterns["phenotypes"], lines[4]):
            print(4)
            return False
        if lines[5] != '~~~':
            print(5)
            return False
        if not re.match(patterns["genes"], lines[6]):
            print(6)
            return False
        if lines[7] != '~~~':
            print(7)
            return False
        if "What" not in lines[8]:
            print(8)
            return False
    except IndexError:
        return False  # Handle cases with missing lines

    return True

def separate_texts(text):
    return text.split('~~~')

