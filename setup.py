from setuptools import setup

pkg_version = 'UNKNOWN'
with open("./torch_flops/__init__.py") as f:
    for line in f.readlines():
        if line.startswith('__version__'):
            pkg_version = line.strip('\n').split('\'')[-2]
readme_path = "README.md"
setup(
    name="torch_flops",
    version=pkg_version,
    author="Yue Lu",
    author_email="luyue163@126.com",
    description="A libiary for calcuating the FLOPs in the forward() process based on torch.fx",
    long_description=open(readme_path, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/zugexiaodui/torch_flops",
    data_files=[readme_path]
)
