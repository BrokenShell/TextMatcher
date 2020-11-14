from setuptools import setup


with open("README.md", "r") as file:
    long_description = file.read()

dev_status = {
    "Alpha": "Development Status :: 3 - Alpha",
    "Beta": "Development Status :: 4 - Beta",
    "Pro": "Development Status :: 5 - Production/Stable",
    "Mature": "Development Status :: 6 - Mature",
}

setup(
    name="NLPTextMatcher",
    author="Robert Sharp",
    author_email="webmaster@sharpdesigndigital.com",
    packages=["NLPTextMatcher"],
    version="0.2.1",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Free for non-commercial use",
    classifiers=[
        dev_status["Beta"],
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "TextMatcher", "natural language processing (NLP)",
        "machine learning (ML)",
    ],
    python_requires='>=3.6',
)
