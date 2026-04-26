# generalsio_copy
An imitation of custom-room mode of generals.io, written in python with flask_socketio.

This imitation only guarantees that the rule is almost the same (I'm not sure if there's some differences) as the original ones, but not the APIs.

Now also supports replay.

## Running
```shell
pip3 install -r requirements.txt
python3 server.py
```

Then just open `http://localhost:23333`.

## Base URL
When deploying behind a reverse proxy that forwards a path prefix, set `GENERALS_BASE_URL`:

```shell
GENERALS_BASE_URL=/generals python3 server.py
```

Then open `http://localhost:23333/generals/`. `BASE_URL` is also accepted as a fallback.
