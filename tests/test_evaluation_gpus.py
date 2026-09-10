import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from evaluation.utils import common, classification, imagenet, distributed_features


def distributed_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=30))
    try:
        # Force multiple transfer chunks and exercise ranks with zero images.
        distributed_features.MAX_TRANSFER_BYTES = 24
        for count in (2, 7):
            expected = torch.arange(count * 6, dtype=torch.float32).reshape(count, 2, 3)
            labels = torch.arange(count * 4, dtype=torch.uint8).reshape(count, 2, 2)
            local, target = expected[rank::world_size], labels[rank::world_size]
            actual, actual_labels = distributed_features.gather_image_shards(
                local if len(local) else None, target if len(local) else None, count,
            )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_labels, labels)
        # Global k-NN accuracy must match the serial result, including a rank
        # with no validation queries. Every training feature remains searchable.
        features = torch.eye(4)
        train_labels = torch.arange(4)
        for count in (2, 4):
            result = imagenet._weighted_knn(features, train_labels, features[:count],
                                           train_labels[:count], neighbors=1, num_classes=4)
            assert result == {'top1': 100., 'top5': 100.}, result
    finally:
        dist.destroy_process_group()


class AutomaticGPUTest(unittest.TestCase):
    def test_launches_all_visible_devices_including_one(self):
        for count in (1, 2, 3, 8):
            with self.subTest(gpus=count), mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                 mock.patch.object(torch.cuda, 'device_count', return_value=count), \
                 mock.patch.object(common.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as launch:
                with self.assertRaises(SystemExit) as stopped:
                    common.launch_distributed_if_needed('evaluation.utils.imagenet_knn')
                self.assertEqual(stopped.exception.code, 0)
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index('--nproc_per_node') + 1], str(count))

    def test_no_gpu_fails_and_existing_torchrun_is_not_relaunched(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'at least one'):
                common.launch_distributed_if_needed('worker')
        with mock.patch.dict(os.environ, {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}), \
             mock.patch.object(torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(torch.cuda, 'device_count', return_value=1), \
             mock.patch.object(common.subprocess, 'run') as launch:
            common.launch_distributed_if_needed('worker')
            launch.assert_not_called()

    def test_single_gpu_initializes_process_group_for_collectives_and_ddp(self):
        with mock.patch.dict(os.environ, {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}), \
             mock.patch.object(torch.cuda, 'set_device') as select, \
             mock.patch.object(dist, 'init_process_group') as initialize, \
             mock.patch.object(dist, 'is_initialized', return_value=True), \
             mock.patch.object(dist, 'get_rank', return_value=0):
            self.assertEqual(common.initialize_distributed(), (0, 1))
            initialize.assert_called_once()
            select.assert_called_once_with(0)

    def test_classification_batch_preserves_reference_for_one_two_four_eight(self):
        for count in (1, 2, 4, 8):
            protocol = classification._protocol('imagenet', 'vit_small', 'teacher', count)
            self.assertEqual(protocol['gpu_count'], count)
            self.assertEqual(protocol['global_batch_size'], 1024)
            self.assertEqual(protocol['batch_size_per_gpu'], 1024 // count)
        self.assertEqual(classification._protocol('imagenet', 'vit_small', 'teacher', 3)['global_batch_size'], 1023)

    def test_feature_microbatching_preserves_head_gradient(self):
        class Backbone(torch.nn.Module):
            def get_intermediate_layers(self, images, n):
                return [images[:, None].expand(-1, 2, -1)] * n
        x, targets = torch.randn(9, 3), torch.arange(9) % 2
        first, second = torch.nn.Linear(12, 2), torch.nn.Linear(12, 2)
        second.load_state_dict(first.state_dict())
        for head, limit in ((first, 256), (second, 2)):
            optimizer = torch.optim.SGD(head.parameters(), lr=.01)
            with mock.patch.object(classification, 'BATCH_SIZE_PER_GPU', limit):
                classification.train_epoch(Backbone(), head, optimizer, [(x, targets)],
                                           'vit_small', False, 'cpu', 0, 0)
        torch.testing.assert_close(first.weight, second.weight)
        torch.testing.assert_close(first.bias, second.bias)

    def test_distributed_shard_order_and_knn_scores_with_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(distributed_worker,
                args=(3, str(Path(directory) / 'store')), nprocs=3, start_method='fork', join=True)


if __name__ == '__main__':
    unittest.main()
