try: from setuptools import setup
except: from distutils.core import setup

setup(	long_description="Automated map georeferencing", #open("README.rst").read(), 
	name="""AutoMap""",
	license="""MIT""",
	author="""Karim Bahgat""",
	author_email="""karim.bahgat.norway@gmail.com""",
	url="""http://github.com/karimbahgat/AutoMap""",
	version="""0.2.0""",
	keywords="""map georeferencing""",
	packages=['automap'],
    requires=['Pillow', 'numpy', 'opencv-python', 'colormath', 'pytesseract'],
	dependency_links=['https://github.com/karimbahgat/PythonGis/tarball/master'],
	classifiers=['License :: OSI Approved', 'Programming Language :: Python', 'Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Intended Audience :: Science/Research', 'Intended Audience :: End Users/Desktop'],
	description="""Automated map georeferencing""",
	)
