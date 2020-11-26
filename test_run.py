import uproot

f = uproot.open("test/testdata/test.root")

test_tree = f['deepntuplizer/tree']
