import sys
import setuptools

if sys.version_info < (3, 7):
    sys.exit('Python>=3.7 is required by TrojPrompt.')

setuptools.setup(
    name="rl_prompt",
    version='0.1.0',
    author=("Jiaqi Xue, Qian Lou"),
    description="TrojPrompt",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='RL Prompt',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=open("requirements.txt", "r").read().split(),
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)