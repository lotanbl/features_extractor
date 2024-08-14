import setuptools
import os

if os.environ.get('GITHUB_REF_NAME') and os.environ.get('GITHUB_REF_NAME') != 'master':
    branch_name='-'+os.environ['GITHUB_REF_NAME'].replace('/','-')
else:
    branch_name=''

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# If you need to add non .py files to your package dist, see this:
# https://python-packaging.readthedocs.io/en/latest/non-code-files.html
requierments = [
    "pandas",
    "emoji==1.7",
    "scikit-learn"
]
setuptools.setup(
    name=f"features-extractor",
    version="0.0.0",
    author="lotan",
    author_email="lotan.levy@perception-point.io",
    description="Package for features extraction and encoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lotanbl/features_extractor",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requierments
)
