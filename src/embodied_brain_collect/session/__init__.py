"""record.session -- session-level orchestration.

   config    Per-session settings (subject, run, output dir, recorder list).
   launcher  Spawn sync_hub + every enabled recorder as subprocesses.
   aligner   Offline alignment: pull every .npz into a single timeline.
"""
