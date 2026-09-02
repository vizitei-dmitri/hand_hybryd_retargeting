from unittest.mock import Mock

from rclpy.serialization import serialize_message
from ros_tcp_endpoint.publisher import RosPublisher
from std_msgs.msg import String


def test_endpoint_deserializes_cdr_before_publishing():
    publisher = RosPublisher.__new__(RosPublisher)
    publisher.msg = String()
    publisher.pub = Mock()

    publisher.send(serialize_message(String(data="from Unity")))

    published = publisher.pub.publish.call_args.args[0]
    assert isinstance(published, String)
    assert published.data == "from Unity"


def test_endpoint_drops_an_invalid_transition_frame_without_disconnect():
    publisher = RosPublisher.__new__(RosPublisher)
    publisher.msg = String()
    publisher.pub = Mock(topic_name="/test")
    logger = Mock()
    publisher.get_logger = Mock(return_value=logger)

    assert publisher.send(b"not ROS 2 CDR") is None

    publisher.pub.publish.assert_not_called()
    logger.warning.assert_called_once()
