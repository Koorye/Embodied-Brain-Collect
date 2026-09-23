import socket, time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tags = [("RUN_START", 241), ("FIX_ON", 17), ("CUE_AUDIO", 33),
        ("GO_AUDIO", 65), ("EXEC_START", 81), ("EXEC_END", 82),
        ("RUN_END", 242)]

print("sending 50 markers to udp://127.0.0.1:9999 (Ctrl+C to abort)")
for i in range(50):
    tag, code = tags[i % len(tags)]
    pkt = (f"EVT|trial=1|tag={tag}|code={code}"
           f"|t_eprime_ms={i*100}|t_sent_pc={time.time()!r}").encode()
    s.sendto(pkt, ("127.0.0.1", 9999))
    print(f"  #{i:02d}  {tag:<10s} code={code}")
    time.sleep(0.3)
print("done.")