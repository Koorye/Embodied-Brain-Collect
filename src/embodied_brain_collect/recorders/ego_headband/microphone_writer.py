"""Validate streamed PCM and write WAV without buffering the whole recording."""
from pathlib import Path
import wave


class MicrophoneWriter:
    RATE = 16000
    CHANNELS = 1
    WIDTH = 2
    SAMPLES = 320

    def __init__(self, path: Path | None):
        self.path = path
        self._file = None
        self._wav = None
        self.next_index = None
        self.next_seq = None
        self.samples_written = 0
        self.closed = False

    def write(self, meta: dict, payload: bytes) -> dict:
        if self.closed:
            raise ValueError('microphone writer already closed')
        audio = meta.get('audio') or {}
        if meta.get('type') != 'audio/PCM' or (
            audio.get('sample_rate'), audio.get('channels'),
            audio.get('format'), audio.get('samples')
        ) != (self.RATE, self.CHANNELS, 'S16_LE', self.SAMPLES):
            raise ValueError('unsupported microphone PCM format')
        if len(payload) != self.SAMPLES * self.WIDTH:
            raise ValueError('microphone PCM block length mismatch')
        index, seq, read_ns = (audio.get('sample_index'), meta.get('stream_seq'),
                               meta.get('read_complete_ns'))
        if any(type(v) is not int for v in (index, seq, read_ns)) or index < 0 or seq < 1 or read_ns <= 0:
            raise ValueError('invalid microphone sample index, sequence, or read timestamp')
        if meta.get('timestamp_basis') != 'read_complete_not_hardware_capture':
            raise ValueError('unsupported microphone timestamp basis')
        # First recorded index can be nonzero because pre-roll is discarded.
        if self.next_index is not None and index != self.next_index:
            raise ValueError(f'microphone sample discontinuity: expected {self.next_index}, got {index}')
        if self.next_seq is not None and seq != self.next_seq:
            raise ValueError(f'microphone packet discontinuity: expected {self.next_seq}, got {seq}')
        if self._wav is None and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open('xb')  # Never overwrite an existing take.
            try:
                self._wav = wave.open(self._file, 'wb')
                self._wav.setparams((self.CHANNELS, self.WIDTH, self.RATE, 0, 'NONE', 'not compressed'))
            except Exception:
                self._file.close()
                raise
        offset = self.samples_written
        if self._wav is not None:
            self._wav.writeframesraw(payload)
        self.samples_written += self.SAMPLES
        self.next_index = index + self.SAMPLES
        self.next_seq = seq + 1
        return {'microphone_read_complete_ns': read_ns,
                'microphone_sample_index': index,
                'microphone_wav_sample_offset': offset,
                'microphone_samples': self.SAMPLES,
                'microphone_stream_seq': seq}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self._wav is not None:
                self._wav.close()
        finally:
            if self._file is not None:
                self._file.close()
