# Transfer Learning

To leverage transfer learning comment out the line "Transfer learning here" and link the desired pretrained dcp model.
Additionally, change in your config file (e.g "./configs/dcp/defaul.yaml") the "NET.POINTER" to either "transformer" or "identity"

dcp-v1 uses the identity
dcp-v2 uses a transformer