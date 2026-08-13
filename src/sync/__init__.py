"""record.sync -- the marker bus.

Two pieces:

  marker_codes  - the single source of truth for the LPT-byte / tag dictionary
                  shared by E-Prime InLine code, sync_hub.py and the recorders.
  sync_hub      - the always-on Python process that ingests E-Prime UDP packets,
                  stamps them with the authoritative PC clock, optionally also
                  forwards them to Pupil Labs Neon (so they are baked into the
                  Neon recording with native nanosecond timestamps), persists
                  them to markers.npz, and rebroadcasts them on a local ZMQ
                  PUB socket so every recorder process can copy them into its
                  own .npz.

Recorders never import sync_hub directly.  They only need
``record.sync.marker_codes`` plus a ``zmq.SUB`` socket on
``tcp://127.0.0.1:9998`` (see record.recorders.base.MarkerSubscriber).

We deliberately don't eager-import submodules here; that triggers a
RuntimeWarning when running ``python -m record.sync.marker_codes`` (it
finds the module already loaded before script execution).
"""
