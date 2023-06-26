import numpy as np
import rospy
from rosbag_to_dataset.dtypes.base import Dtype
from vesc_msgs.msg import VescStateStamped


class VescStateConvert(Dtype):
    """
    Convert an VescStateStamped observation into speed
    """

    def __init__(self):
        return

    def N(self):
        return 1

    def rosmsg_type(self):
        return VescStateStamped

    def ros_to_numpy(self, msg):
        return np.array([msg.state.speed])


if __name__ == "__main__":
    c = VescStateConvert()
    msg = VescStateStamped()

    print(c.ros_to_numpy(msg))
