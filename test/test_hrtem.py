from hypothesis import given
from hypothesis.strategies import integers, floats, composite,booleans
from abtem import PlaneWave


@composite
def plane_wave(draw,
               gpts=integers(min_value=1, max_value=1024),
               sampling=floats(min_value=.01, max_value=1),
               energy=floats(min_value=50e3, max_value=500e3),
               normalize=booleans()
               ):
    gpts = draw(gpts)
    sampling = draw(sampling)
    energy = draw(energy)
    normalize = draw(normalize)
    plane_wave = PlaneWave(gpts=gpts, sampling=sampling, energy=energy, normalize=normalize)

    return plane_wave


@given(plane_wave=plane_wave())
def test_decode_inverts_encode(plane_wave):

    plane_wave.build()

    print(plane_wave.wavelength)
    #assert isinstance(plane_wave, int)




# @composite
# def list_and_index(draw, elements=integers()):
#     xs = draw(lists(elements, min_size=1))
#     i = draw(integers(min_value=0, max_value=len(xs) - 1))
#     return (xs, i)

#
# @composite
# def plane_wave(gpts=integers()):
#
#     gpts = gpts
#
#     return gpts
#
#
#
# @given(plane_wave=plane_wave())
# def test_decode_inverts_encode(plane_wave):
#     assert isinstance(plane_wave, int)
#


# import hypothesis.strategies as st
#
# from hypothesis import given
#
# @st.composite
# def s(draw, gpts=integers()):
#     x = draw(st.text())
#
#     gpts =draw(gpts)
#     #y = draw(st.text(alphabet=x))
#     return (x, gpts)
#
# @given(s1=s(), s2=s())
# def test_subtraction(s1, s2):
#
#     print(s1, s2)
#
#     #assert 0