#scripts/run_loso.py
import os
import subprocess

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
subjects = list(range(1, 10))

for test_sid in subjects:

    train_subjects = [
        s for s in subjects
        if s != test_sid
    ]

    train_str = "[" + ",".join(map(str, train_subjects)) + "]"
    test_str = "[" + str(test_sid) + "]"

    save_dir = os.path.join(workspace_root, 'outputs', f'subject{test_sid}')

    cmd = (
        f"python train.py "
        f"experiment.train_subjects={train_str} "
        f"experiment.test_subjects={test_str} "
        f"train.save_dir=\"{save_dir}\""
    )

    print("=" * 80)
    print(f"TEST SUBJECT = {test_sid}")
    print(cmd)

    subprocess.run(cmd, shell=True)