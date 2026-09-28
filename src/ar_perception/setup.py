from setuptools import find_packages, setup

package_name = 'ar_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='td',
    maintainer_email='kangsy060125@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
          'scan_test_node = ar_perception.scan_test_node:main',
          'scan_xy_test_node = ar_perception.scan_xy_test_node:main',
          'lidar_preprocess_node = ar_perception.lidar_preprocess_node:main',
          'lidar_perception = ar_perception.lidar_perception:main',
        ],
    },
)
