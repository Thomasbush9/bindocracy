"""Independent campaign controllers serialize access to one DuckDB file."""
import multiprocessing

from bindocracy.store.locking import locked_connection


def _read_after_writer(database, ready, results):
    ready.set()
    with locked_connection(database, read_only=True) as connection:
        results.send(connection.execute("SELECT value FROM observations").fetchone()[0])
    results.close()


def test_reader_waits_for_writer_and_observes_committed_result(tmp_path):
    database = tmp_path / "campaign.duckdb"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_read_after_writer, args=(database, ready, sender))
    try:
        with locked_connection(database) as connection:
            connection.execute("CREATE TABLE observations(value INTEGER)")
            connection.execute("INSERT INTO observations VALUES (17)")
            child.start()
            sender.close()
            assert ready.wait(20), "reader did not reach database acquisition"
            assert not receiver.poll(), "reader bypassed the active writer"
            connection.execute("UPDATE observations SET value=23")
        assert receiver.poll(20), "reader did not resume after writer closed"
        assert receiver.recv() == 23
        child.join(20)
        assert child.exitcode == 0
    finally:
        if child.pid is not None and child.is_alive():
            child.terminate()
            child.join(5)
        receiver.close()
        sender.close()
