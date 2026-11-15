"""DDSを起動せず、プロセス内の接続元設定と初期化前の条件を検証する。"""
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_chat"))
from dds_network import configure_dds_address


CONFIG = '''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS><Domain Id="any"><General><Interfaces>
<NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
</Interfaces></General><Tracing><Verbosity>config</Verbosity>
<OutputFile>/tmp/cdds.LOG</OutputFile></Tracing></Domain></CycloneDDS>'''


def fake_channel(config=CONFIG, initialized=False, with_lock=True):
    fields = {"_ChannelFactory__initialized": initialized}
    if with_lock:
        fields["_ChannelFactory__init_lock"] = threading.Lock()
    factory = type("FakeFactory", (), fields)
    return types.SimpleNamespace(ChannelFactory=factory, ChannelConfigHasInterface=config)


class DDSAddressTests(unittest.TestCase):
    def test_unspecified_is_noop_without_import_even_after_initialization(self):
        channel = fake_channel(initialized=True)
        with mock.patch("dds_network.importlib.import_module") as imported:
            for address in (None, ""):
                self.assertIsNone(configure_dds_address(address))
                self.assertIsNone(configure_dds_address(address, channel))
        imported.assert_not_called()
        self.assertEqual(channel.ChannelConfigHasInterface, CONFIG)

    def test_explicit_address_replaces_only_the_interface_name_attribute(self):
        channel = fake_channel()
        with mock.patch("dds_network.importlib.import_module") as imported:
            result = configure_dds_address("192.168.123.164", channel)
        imported.assert_not_called()
        self.assertEqual(result, "192.168.123.164")
        self.assertEqual(channel.ChannelConfigHasInterface,
                         CONFIG.replace('name="$__IF_NAME__$"', 'address="192.168.123.164"'))
        self.assertNotIn('name="', channel.ChannelConfigHasInterface)
        self.assertFalse(channel.ChannelFactory._ChannelFactory__initialized)

    def test_valid_address_imports_only_the_requested_sdk_module(self):
        channel = fake_channel()
        with mock.patch("dds_network.importlib.import_module", return_value=channel) as imported:
            self.assertEqual(configure_dds_address("192.168.123.164"), "192.168.123.164")
        imported.assert_called_once_with("unitree_sdk2py.core.channel")

    def test_invalid_addresses_fail_before_sdk_import_or_mutation(self):
        invalid = (False, True, 1, b"192.168.123.164", [], {}, "not-an-ip",
                   "192.168.123.999", "192.168.123.0164", "192.168.123.164/24",
                   " 192.168.123.164 ", "::1", "::ffff:192.168.123.164",
                   "0.0.0.0", "224.0.0.1", "239.255.0.1", "255.255.255.255",
                   '192.168.123.164" name="Ethernet')
        with mock.patch("dds_network.importlib.import_module") as imported:
            for address in invalid:
                channel = fake_channel()
                with self.subTest(address=address), self.assertRaises(ValueError):
                    configure_dds_address(address)
                with self.subTest(address=address), self.assertRaises(ValueError):
                    configure_dds_address(address, channel)
                self.assertEqual(channel.ChannelConfigHasInterface, CONFIG)
        imported.assert_not_called()

    def test_loopback_is_valid_for_explicit_virtual_binding(self):
        channel = fake_channel()
        self.assertEqual(configure_dds_address("127.0.0.1", channel), "127.0.0.1")
        self.assertIn('address="127.0.0.1"', channel.ChannelConfigHasInterface)

    def test_initialized_sdk_cannot_be_reconfigured(self):
        channel = fake_channel(initialized=True)
        with self.assertRaisesRegex(RuntimeError, "初期化後"):
            configure_dds_address("192.168.123.164", channel)
        self.assertEqual(channel.ChannelConfigHasInterface, CONFIG)

    def test_missing_duplicate_or_unexpected_template_is_not_mutated(self):
        for config in (None, 123, CONFIG.replace('name="$__IF_NAME__$"', 'name="Ethernet"'),
                       CONFIG + ' name="$__IF_NAME__$"'):
            channel = fake_channel(config=config)
            with self.subTest(config=config), self.assertRaisesRegex(RuntimeError, "設定形式"):
                configure_dds_address("192.168.123.164", channel)
            self.assertEqual(channel.ChannelConfigHasInterface, config)

    def test_missing_factory_is_rejected_without_mutating_template(self):
        channel = types.SimpleNamespace(ChannelConfigHasInterface=CONFIG)
        with self.assertRaisesRegex(RuntimeError, "ChannelFactory"):
            configure_dds_address("192.168.123.164", channel)
        self.assertEqual(channel.ChannelConfigHasInterface, CONFIG)

    def test_noop_does_not_restore_an_already_selected_template(self):
        channel = fake_channel()
        configure_dds_address("192.168.123.164", channel)
        changed = channel.ChannelConfigHasInterface
        self.assertIsNone(configure_dds_address(None, channel))
        self.assertIsNone(configure_dds_address("", channel))
        self.assertEqual(channel.ChannelConfigHasInterface, changed)

    def test_configuration_uses_the_sdk_initialization_lock(self):
        channel = fake_channel()
        lock = channel.ChannelFactory._ChannelFactory__init_lock
        entered, finished = threading.Event(), threading.Event()
        errors = []

        def configure():
            entered.set()
            try:
                configure_dds_address("192.168.123.164", channel)
            except RuntimeError as error:
                errors.append(str(error))
            finally:
                finished.set()

        lock.acquire()
        worker = threading.Thread(target=configure)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(finished.wait(.02))
            channel.ChannelFactory._ChannelFactory__initialized = True
        finally:
            lock.release()
            worker.join(1)
        self.assertTrue(finished.is_set())
        self.assertEqual(len(errors), 1)
        self.assertEqual(channel.ChannelConfigHasInterface, CONFIG)

    def test_minimal_fake_without_initialization_lock_can_be_validated(self):
        channel = fake_channel(with_lock=False)
        self.assertEqual(configure_dds_address("192.168.123.164", channel), "192.168.123.164")


if __name__ == "__main__":
    unittest.main()
