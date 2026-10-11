# Previous playing-model restoration

Selected: preserved Geometric V13, SHA256 `55fae5177140dc3142514ebaf08c89c9c39263b5e8c105c0c9d3d811df1121dc`.
Replaced playing entry: Outcome-128, SHA256 `8bf824f43c295bf397e443f019940c00cb7f3e0ec3b61da13b274d30e8211043`.
Protected previous main: `738e875877bec6ce84965a607cc15af2341f9a87`.

The user requested restoring the previous best bot and delegated the model choice. Geometric was selected because it is the last version the user personally reported slightly better than Original V13 in native gameplay. Outcome's latest native feedback is unfavorable; Ground was not accepted locally. No new claim of improved model strength is made.

The new commit is built on previous main and changes the playing checkpoint through a normal Git commit, preserving full history. No force push, checkpoint deletion or new training. All preserved models and private backup sets remain. The original decoder, observations, recurrent semantics, architecture and callback timing stay intact. The only inference code modification is the necessary pinned checkpoint SHA; bot/preflight identity literals match the restored launcher identity. Logging remains diagnostic only.

Verification includes strict parameter/tensor loading, exact original runtime inference parity on chronological real-state observations, original action mapping/SDK preflight, sealing and the untouched official validator. Exact results are recorded separately. Private ZIP upload is downloaded again, checksum verified and cold restored before reporting.
