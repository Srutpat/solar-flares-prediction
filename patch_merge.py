from pathlib import Path
import shutil

p = Path(r"src\data_adapter\adapter.py")
shutil.copy(p, p.with_suffix(".py.bak2"))
s = p.read_text(encoding="utf-8")

def rep(old, new):
    global s
    assert s.count(old) == 1, f"anchor not unique/found:\n{old}"
    s = s.replace(old, new)

I = " " * 20  # indent inside the loop

# 1) unique id per archive while building
rep(
    "stream_id = self._stream_id(instrument, detector, ext, loaded.ref.display_name)\n",
    "base_stream_id = self._stream_id(instrument, detector, ext, loaded.ref.display_name)\n"
    + I + 'archive_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(loaded.ref.source).stem)\n'
    + I + 'stream_id = f"{base_stream_id}__{archive_tag}"\n',
)

# 2) remember the base id in metadata
rep(
    'meta["time_column"] = time_col\n',
    'meta["time_column"] = time_col\n'
    + I + 'meta["base_stream_id"] = base_stream_id\n'
    + I + 'meta["archive_source"] = loaded.ref.source\n',
)

# 3) merge before returning
rep(
    "        report.n_observations = int(sum(len(s.dataframe) for s in streams.values()))\n        return streams\n",
    "        streams = self._merge_archive_streams(streams)\n"
    "        report.n_observations = int(sum(len(s.dataframe) for s in streams.values()))\n"
    "        return streams\n",
)

# 4) the merge function itself
rep(
    "    @staticmethod\n    def _stream_id(instrument, detector, extension, source_file) -> str:\n",
    '''    @staticmethod
    def _merge_archive_streams(streams):
        """Merge same detector/band streams from split archives (e.g. two HEL1OS halves of a day).
        Data is concatenated chronologically; GTIs are the union of every archive's own GTIs."""
        groups: dict[str, list] = {}
        for st in streams.values():
            groups.setdefault(st.metadata.get("base_stream_id", st.stream_id), []).append(st)
        merged = {}
        for base_id, parts in groups.items():
            if len(parts) == 1:
                st = parts[0]
                st.stream_id = base_id
                merged[base_id] = st
                continue
            parts.sort(key=lambda q: q.dataframe["timestamp"].min())
            frame = (
                pd.concat([q.dataframe for q in parts], ignore_index=True)
                .sort_values("timestamp", kind="stable")
                .drop_duplicates("timestamp", keep="first")
                .reset_index(drop=True)
            )
            meta = dict(parts[0].metadata)
            gtis = sorted({tuple(i) for q in parts for i in q.metadata.get("gti_intervals_utc", [])})
            meta["gti_intervals_utc"] = gtis
            meta["gti_source"] = [g for q in parts for g in q.metadata.get("gti_source", [])]
            meta["gti_status"] = (
                "available" if any(q.metadata.get("gti_status") == "available" for q in parts)
                else parts[0].metadata.get("gti_status", "not_found")
            )
            meta["merged_from"] = [q.source_file for q in parts]
            first = parts[0]
            merged[base_id] = StandardizedStream(
                stream_id=base_id,
                instrument=first.instrument,
                detector=first.detector,
                product_type=first.product_type,
                extension=first.extension,
                source_file=" + ".join(q.source_file for q in parts),
                measurement_columns=first.measurement_columns,
                dataframe=frame,
                metadata=meta,
            )
        return merged

    @staticmethod
    def _stream_id(instrument, detector, extension, source_file) -> str:
''',
)

p.write_text(s, encoding="utf-8")
print("patched OK")