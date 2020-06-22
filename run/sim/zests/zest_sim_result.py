from zest import zest


@zest.skip("m", "Manas")
def zest_sim_result():
    def it_gets_flat():
        raise NotImplementedError

    def unflat():
        raise NotImplementedError

    def it_gets_flus():
        raise NotImplementedError

    def it_gets_peps__flus():
        raise NotImplementedError

    def it_gets_peps__flus__unique_flus():
        raise NotImplementedError

    def it_gets_pros__peps__pepstrs__flus():
        raise NotImplementedError

    zest()
