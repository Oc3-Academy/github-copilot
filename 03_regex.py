import re

# regex to match a string that starts with 'a' and ends with 'b'
regex = re.compile(r"^a.*b$")

# regex to match a string that starts with 'a' and ends with 'b' and has 'c' in between
regex = re.compile(r"^a.*c.*b$")

# regex to match numbers in a string
regex = re.compile(r"\d+")

# regex to match a email address in a string
regex = re.compile(r"[\w\.-]+@[\w\.-]+")

r"""(\d{3}|\(\d{3}\))(:?\s+|-|\.)?(\d{3})(:?\s+|-|\.)?(\d{4})"""
# what patterns does this regex match?
# 1234567890
# (123)4567890
# (123) 4567890
# (123) 456 7890
# 123 456 7890
# 123-456-7890
