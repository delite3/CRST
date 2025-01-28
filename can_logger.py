from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Union
import can
import cantools
import numpy as np
import numpy.typing as npt


class CanLogger:
    def __init__(
        self, dbc_path: Path, log_path: Path, time_between_msgs: timedelta
    ) -> None:
        # Saves simulated radar data as a .blf CAN log using frame and message
        # definitions from the CAS project

        self.db = cantools.db.load_file(dbc_path)
        assert isinstance(self.db, cantools.database.can.database.Database)

        # Create a virtual bus for loopback
        self.bus = can.Bus(
            interface="virtual",
            channel="channel_0",
            receive_own_messages=True,
            preserve_timestamps=True,
        )

        # Setup a logger
        self.logger = can.Logger(log_path)

        # Connect the logger to the virtual bus. Any messages sent on the bus
        # will be logged
        self.notifier = can.Notifier(bus=self.bus, listeners=[self.logger])

        # Simulated delay to add between each message on the bus
        self.time_between_msgs = time_between_msgs

        # CAN messages and frame ids are stored here to simplify the logging code.
        # Prefill dicts with dummy data for each message, s.t. we only need to modify
        # the signal of interest when the logging function is called.

        # OXTS velocity message
        self.vel_frame_id = 0x603
        self.vel_msg = self.db.get_message_by_frame_id(self.vel_frame_id)
        self.vel_dict: Dict[str, float] = {
            "VelNorth": 0,
            "VelEast": 0,
            "VelDown": 0,
            "Speed2D": 0,
        }

        # IMU yaw rate message
        self.yaw_rate_frame_id = 0x174
        self.yaw_rate_msg = self.db.get_message_by_frame_id(self.yaw_rate_frame_id)
        self.yaw_rate_dict: Dict[str, Union[int, float]] = {
            "YAW_RATE": 0,
            "CLU_STAT": 0,
            "YAW_RATE_STAT": 0,
            "TEMP_RATE_Z": 0,
            "AY": 0,
            "MSG_CNT": 0,
            "AY_STAT": 0,
            "CRC": 0,
        }

        # Radar messages
        self.n_radar_msgs = 48
        self.first_radar_frame_id = 0x200
        self.last_radar_frame_id = 0x25F
        assert (
            self.n_radar_msgs
            == (self.last_radar_frame_id - self.first_radar_frame_id + 1) // 2
        )
        # A messages
        self.radar_a_msgs = [
            self.db.get_message_by_frame_id(frame_id)
            for frame_id in range(
                self.first_radar_frame_id, self.last_radar_frame_id, 2
            )
        ]
        # B messages
        self.radar_b_msgs = [
            self.db.get_message_by_frame_id(frame_id)
            for frame_id in range(
                self.first_radar_frame_id + 1, self.last_radar_frame_id + 1, 2
            )
        ]
        # A message dict, note that "AMessage" must be set to 1.
        self.radar_data_a_dict: Dict[str, Union[int, float]] = {
            "measured": 0,
            "AMessage": 1,
            "CRC": 0,
            "MC": 0,
            "ID": 0,
            "vr": 0,
            "dr": 0,
            "phi": 0,
            "dbPower": 0,
        }
        # B message dict
        self.radar_data_b_dict: Dict[str, Union[int, float]] = {
            "AMessage": 0,
            "pExist": 0,
            "timeSinceMeas": 0,
            "CRC": 0,
            "MC": 0,
            "ID": 0,
            "drSdv": 0,
            "vrSdv": 0,
            "phiSdv": 0,
        }

        # As the radar always sends all 48 messages, we must do that as well. It's
        # convenient to pre-encode empty A and B messages here (with "measured" set to 0),
        # and just send these over and over.
        self.empty_radar_data_a = self.radar_a_msgs[0].encode(self.radar_data_a_dict)
        self.empty_radar_data_b = self.radar_b_msgs[0].encode(self.radar_data_b_dict)

    def sort_by_snr(
        self,
        r: npt.NDArray[np.float64],
        vr: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        snr: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        # Sorts the detections by SNR
        sort_idx = np.argsort(snr)
        r = r[sort_idx]
        vr = vr[sort_idx]
        phi = phi[sort_idx]
        snr = snr[sort_idx]
        return r, vr, phi, snr

    def send(self, data: bytes, frame_id: int, t: datetime) -> datetime:
        # Send the encoded data on the bus, increment the time
        msg = can.Message(timestamp=t.timestamp(), arbitration_id=frame_id, data=data)
        self.bus.send(msg)
        return t + self.time_between_msgs

    def log(
        self,
        t: datetime,
        speed: float,
        yaw_rate: float,
        r: npt.NDArray[np.float64],
        vr: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        snr: npt.NDArray[np.float64],
        rcs: npt.NDArray[np.float64],
    ) -> None:
        # Log detections and ego data
        # The time t is used to simulate a received time in the CAN messages,
        # it's incremented each time a message is sent to the bus.
        t = self.log_radar(r, vr, phi, snr, rcs, t)
        t = self.log_ego(speed, yaw_rate, t)

    def log_ego(self, speed: float, yaw_rate: float, t: datetime) -> datetime:
        # Encode the ego data and send it on the bus

        self.vel_dict["Speed2D"] = speed
        speed_data = self.vel_msg.encode(self.vel_dict)
        t = self.send(speed_data, self.vel_frame_id, t)

        self.yaw_rate_dict["YAW_RATE"] = yaw_rate
        yaw_rate_data = self.yaw_rate_msg.encode(self.yaw_rate_dict)
        t = self.send(yaw_rate_data, self.yaw_rate_frame_id, t)

        return t

    def log_radar(
        self,
        r: npt.NDArray[np.float64],
        vr: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        snr: npt.NDArray[np.float64],
        rcs: npt.NDArray[np.float64],
        t: datetime,
    ) -> datetime:
        # Encode the radar data and send it on the bus

        r, vr, phi, snr = self.sort_by_snr(r, vr, phi, snr)
        n_dets = len(r)

        # Encode and send all A and B messages
        for i in range(self.n_radar_msgs):
            msg_a = self.radar_a_msgs[i]
            frame_id_a = msg_a.frame_id

            msg_b = self.radar_b_msgs[i]
            frame_id_b = msg_b.frame_id

            if i > n_dets - 1:
                # We're out of detections, use empty messages
                data_a = self.empty_radar_data_a
                data_b = self.empty_radar_data_b
            else:
                # Encode detection
                self.radar_data_a_dict["measured"] = 1

                self.radar_data_a_dict["vr"] = vr[i]
                self.radar_data_a_dict["dr"] = r[i]
                self.radar_data_a_dict["phi"] = phi[i]
                self.radar_data_a_dict["dbPower"] = rcs[i]
                data_a = msg_a.encode(self.radar_data_a_dict)
                data_b = msg_b.encode(self.radar_data_b_dict)

            t = self.send(data_a, frame_id_a, t)
            t = self.send(data_b, frame_id_b, t)
        return t

    def close(self) -> None:
        # Stop the logger and let it close the file
        # Must be called for a graceful exit
        self.logger.stop()
