# python -m microscopium.serve tests/testdata/images/data.csv\
#         -c tests/testdata/images/settings.yaml

# python -m microscopium.serve bbbc021_int/meta_test.csv\
#         -c bbbc021_int/settings.yaml

# python -m microscopium.serve bomi/bomi_small.csv \
#         -c bomi/settings.yaml \
#         -P 5001

python -m microscopium.serve_channels bomi/bomi_small.csv\
        -c bomi/settings_5channel.yaml \
        -P 5001