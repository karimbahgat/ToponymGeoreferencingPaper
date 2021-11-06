try: from setuptools import setup
except: from distutils.core import setup

setup(	long_description="Automated map georeferencing", #open("README.rst").read(), 
	name="""AutoMap""",
	license="""MIT""",
	author="""Karim Bahgat""",
	author_email="""karim.bahgat.norway@gmail.com""",
	url="""http://github.com/karimbahgat/AutoMap""",
	version="""0.3.0""",
	keywords="""map georeferencing""",
	packages=['automap'],
    install_requires=['Pillow', 'numpy', 
					'opencv-python-headless==4.3.0.36', # see: https://github.com/opencv/opencv-python/issues/370
					'colormath', 'pytesseract',
					'PythonGIS @ git+https://github.com/karimbahgat/PythonGis'],
	#dependency_links=['http://github.com/karimbahgat/PythonGis/tarball/master#egg=PythonGIS-0.3.0'],
	classifiers=['License :: OSI Approved', 'Programming Language :: Python', 'Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Intended Audience :: Science/Research', 'Intended Audience :: End Users/Desktop'],
	description="""Automated map georeferencing""",
	)
