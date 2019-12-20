from distutils.core import setup

setup(
    name = 'FortuneCookieGenerator',
    version=0.1,
    description='Fortune Cookie Quote Generator',
    url='https://github.com/WomenInDataScience-Seattle/FortuneCookie',
    packages=['fortune_cookie_generator'],
    install_requires=['tensorflow<2.0.0', 'scipy', 'keras', 'flask'],
    license='',
    author='Thomas George, Brianna Brown, Tony Tao, Dan Dong'
)