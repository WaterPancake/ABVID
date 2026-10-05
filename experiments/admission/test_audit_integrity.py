"""Regression checks for admission failure modes; no dataset/model execution."""
import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audit_integrity import audit_file, base_row, parse_idmt, parse_melaudis, parse_synthetic, quarantine_label_conflicts


class IntegrityChecks(unittest.TestCase):
    def audit(self, samples, subtype='PCM_16', modify=None, channels=1, metadata=None):
        b = io.BytesIO()
        sf.write(b, samples, 8000, format='WAV', subtype=subtype)
        data = b.getvalue()
        if modify:
            data = modify(data)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.wav'
            path.write_bytes(data)
            row = base_row('fixture', 'fixture', 'sample.wav')
            row.update(canonical_class='car', class_id=0, expected_channels=channels)
            row.update(metadata or {})
            return audit_file((row, path))

    def test_full_valid_payload_and_no_admission_or_split(self):
        row = self.audit(np.full(16000, 0.01))
        self.assertTrue(row['integrity_and_class_pass'])
        self.assertEqual(row['decoded_frames'], 16000)
        self.assertFalse(row['admitted_for_training'])
        self.assertIsNone(row['split_role'])

    def test_decoder_tolerated_truncation_is_rejected(self):
        row = self.audit(np.full(16000, 0.1), modify=lambda b: b[:-8])
        self.assertFalse(row['integrity_and_class_pass'])
        self.assertIn('riff_declared_length_exceeds_file', row['integrity_errors'])

    def test_nan_silence_short_and_channel_mismatch(self):
        for x in [np.zeros(16000), np.full(16000, np.nan), np.ones(15999) * 0.1]:
            self.assertFalse(self.audit(x, 'FLOAT')['integrity_and_class_pass'])
        self.assertFalse(self.audit(np.ones(16000)*0.1, channels=2)['integrity_and_class_pass'])

    def test_quiet_and_full_scale_are_not_exclusions(self):
        self.assertTrue(self.audit(np.full(16000, 1e-12), 'FLOAT')['integrity_and_class_pass'])
        row = self.audit(np.ones(16000))
        self.assertTrue(row['integrity_and_class_pass'])
        self.assertIn('near_full_scale_not_proof_of_clipping', row['flags'])

    def test_stereo_mono_cancellation_is_rejected(self):
        x = np.column_stack([np.ones(16000)*0.1, np.ones(16000)*-0.1])
        row = self.audit(x, 'FLOAT', channels=2)
        self.assertIn('all_zero_mono_downmix', row['integrity_errors'])

    def test_decoded_hash_ignores_container_metadata(self):
        def append_junk(b):
            out = bytearray(b + b'JUNK' + struct.pack('<I', 4) + b'abcd')
            struct.pack_into('<I', out, 4, len(out)-8)
            return bytes(out)
        x = np.sin(np.arange(16000) * 0.1) * 0.1
        a, b = self.audit(x), self.audit(x, modify=append_junk)
        self.assertTrue(b['integrity_and_class_pass'])
        self.assertNotEqual(a['source_file_sha256'], b['source_file_sha256'])
        self.assertEqual(a['decoded_audio_sha256'], b['decoded_audio_sha256'])

    def test_redundant_mono_versus_distinct_channels(self):
        a = np.ones(16000)*0.1
        metadata = {'dataset_id':'MELAUDIS','provider_channel_layout':'mono'}
        same = self.audit(np.column_stack([a,a]), metadata=metadata)
        different = self.audit(np.column_stack([a,a*2]), metadata=metadata)
        self.assertTrue(same['integrity_and_class_pass'])
        self.assertIn('mono_label_with_distinct_channels', different['channel_errors'])

    def test_conflicting_labels_quarantine_every_copy(self):
        samples = np.ones(16000)*0.1
        car = self.audit(samples, metadata={'file_id':'a','raw_label':'car'})
        truck = self.audit(samples, metadata={'file_id':'b','raw_label':'truck'})
        self.assertEqual(len(quarantine_label_conflicts([car,truck])), 1)
        self.assertFalse(car['integrity_and_class_pass'])
        self.assertFalse(truck['integrity_and_class_pass'])


class MetadataChecks(unittest.TestCase):
    def test_idmt_pairs_share_event_but_only_se_ch34_passes(self):
        stem = '2019-10-23-16-20_Fraunhofer-IDMT_30Kmh_990828_A_D_CL_'
        se, me = parse_idmt(stem+'SE_CH34.wav'), parse_idmt(stem+'ME_CH12.wav')
        self.assertEqual(se['paired_event_id'], me['paired_event_id'])
        self.assertEqual(se['metadata_exclusion_reasons'], [])
        self.assertIn('sensor_not_selected_ME', me['metadata_exclusion_reasons'])
        self.assertEqual(se['original_channel_ids'], [3, 4])
        self.assertIsNone(se['vehicle_speed'])
        self.assertTrue(parse_idmt(stem+'SE_CH12.wav')['metadata_exclusion_reasons'])

    def test_motorcycle_not_relabelled(self):
        r = parse_idmt('2020-08-29-10-20_Hohenwarte_unknownKmh_123456_M_D_ML_SE_CH12.wav')
        self.assertIsNone(r['canonical_class'])
        self.assertIn('class_out_of_scope_v1.1', r['metadata_exclusion_reasons'])

    def test_melaudis_exact_single_vehicle_and_state(self):
        def row(state, label):
            return parse_melaudis(f'Final_Veh/X/2024-02-09_13-0-1.8-Fitzroy1_{state}_{label}_2Lanes_LR_stereo.wav')
        self.assertEqual(row('FF','1V-Car')['metadata_exclusion_reasons'], [])
        self.assertEqual(row('FF','1V-Truck')['class_id'], 1)
        self.assertTrue(row('FF','2V-TruckCar')['metadata_exclusion_reasons'])
        self.assertTrue(row('TJN','1V-Car')['metadata_exclusion_reasons'])

    def test_synthetic_class_mismatch_fails_and_index_is_not_seed(self):
        a = parse_synthetic('data/synthetic/audioldm/car/ALDM_truck_0001.wav')
        self.assertFalse(a['metadata_parse_ok'])
        b = parse_synthetic('data/synthetic/audioldm/car/ALDM_car_0201.wav')
        self.assertEqual(b['metadata_exclusion_reasons'], [])
        self.assertIsNone(b['source_seed'])

    def test_invalid_time_retains_known_label_and_channels(self):
        r = parse_melaudis('Final_Veh/Hod1/2023-11-08_14-60-1.5-Hoddle1_FF_1V-Car_L1_3Lanes_RL_stereo.wav')
        self.assertEqual(r['canonical_class'], 'car')
        self.assertEqual(r['expected_channels'], 2)
        self.assertIn('invalid_timestamp_token', r['metadata_exclusion_reasons'])
        self.assertEqual(r['timestamp_token'], '14-60-1.5-Hoddle1')


if __name__ == '__main__':
    unittest.main()
