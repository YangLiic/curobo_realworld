from setuptools import setup, find_packages

setup(
    name="single_plan",
    version="0.1.0",
    description="CuroboPlanner class for Isaac Sim + CuRobo integration",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        # 其他依赖可按需添加
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
