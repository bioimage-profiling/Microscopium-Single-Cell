# python -m microscopium.serve tests/testdata/images/data.csv\
#         -c tests/testdata/images/settings.yaml

# python -m microscopium.serve bbbc021_int/meta_test.csv\
#         -c bbbc021_int/settings.yaml

python -m microscopium.serve_channels bomi/bomi_umap.csv\
        -c bomi/settings_5channel.yaml

# python -m microscopium.serve_channels bomi/bomi_meanstd.csv\
#         -c bomi/settings_meanstd.yaml