from setuptools import setup

setup(
    name='gtlo',
    version='0.1',
    keywords='rl, morl, environment, openai-gym, gym, optimal control',
    author="Johannes Dornheim",
    author_email='johannes.dornheim@mailbox.org',
    install_requires=[
        'gym==0.18',
        'pillow>=4',
        'stable-baselines',
    ],
    packages=["gtlo"],
    include_package_data=False
)
