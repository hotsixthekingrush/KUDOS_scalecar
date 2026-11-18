import numpy as np
from ar_perception.wall_filter import WallFilter
from ar_perception.point_accumulation import transform


def line(x0=0.,x1=2.,y=1.,n=100):
    return np.column_stack((np.linspace(x0,x1,n),np.full(n,y)))


def test_wall_and_nearby_vehicle_and_finite_extent():
    f=WallFilter();f.update(1.,None,line())
    assert f.reject(line(.7,.9,n=12))[0]
    assert not f.reject(line(.7,.9,y=.9,n=12))[0]
    assert not f.reject(line(2.1,2.3,n=12))[0]


def test_small_collinear_objects_not_merged_across_gap():
    f=WallFilter();f.update(1.,None,np.vstack([line(0,.2,n=15),line(.7,.9,n=15)]))
    assert not f.walls


def test_tracking_rotation_expiry_and_missing_odom():
    f=WallFilter();f.update(1.,(0,0,0),line())
    pose=(.05,.01,.08);candidate=transform(line(.7,.9,n=12),pose,inverse=True)
    f.update(1.1,pose,[])
    assert f.reject(candidate)==(True,'odom')
    f.update(1.6,pose,[]);assert not f.reject(candidate)[0]
    f.update(2.,(0,0,0),line());f.update(2.1,None,[])
    assert not f.walls and not f.memory


def test_clock_epoch_and_pose_jump_clear():
    for mode in ['clock','epoch','jump']:
        f=WallFilter();f.update(1.,(0,0,0),line())
        f.update(.9 if mode=='clock' else 1.1,(2,0,0) if mode=='jump' else (0,0,0),[],1 if mode=='epoch' else 0)
        assert not f.walls


def test_current_scan_without_odom_and_nonfinite():
    f=WallFilter();f.update(1.,None,np.vstack([line(),[[float('nan'),0]]]))
    assert f.reject(line(.7,.9,n=12))==(True,'scan')
    f.update(float('nan'),None,line());assert not f.walls
