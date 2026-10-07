import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'line_tracing_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KUDOS',
    maintainer_email='tjdus@kookmin.ac.kr',
    description=(
        'KUDOS 스케일카 자율주행 - 카메라 기반 차선 추종(line tracing) 노드'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_tracing_node = line_tracing_pkg.line_tracing_node:main',
            'lab_measure = line_tracing_pkg.lab_measure:main',
        ],
    },
)
