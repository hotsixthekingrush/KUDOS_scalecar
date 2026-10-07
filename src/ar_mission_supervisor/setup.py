import os
from glob import glob
from setuptools import setup

package_name = 'ar_mission_supervisor'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, package_name + '.states'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KUDOS',
    maintainer_email='hsj061008@gmail.com',
    description='AutoRace 판단 모듈 (Supervisor + Planner States)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'supervisor = ar_mission_supervisor.supervisor:main',
            'fake_perception = ar_mission_supervisor.fake_perception:main',
        ],
    },
)
