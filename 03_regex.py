import re

# regex to match a string that starts with 'a' and ends with 'b'
regex = re.compile(r"^a.*b$")

regex2 = re.compile(
    r"""
    (\d{3}|\(\d{3}\))(:?\s+|-|\.)?(\d{3})(:?\s+|-|\.)?(\d{4})
    """
)
# what patterns does this regex match?
# 123-456-7890
# 123.456.7890
# 123 456 7890
# (123)456-7890
# (123) 456-7890
# (123) 456 7890
