from esmi.secs import Securities
from esmi.pricer import Pricer
from esmi.better import Better
import esmi.polymarket as pm

def main():
    secs = Securities()
    p = Pricer()
    b = Better()

    sec_candidates = pm.load_secs(max_pages=1000)
    for sec in sec_candidates:
        secs.create_sec(sec[0], sec[1], sec[2], sec[3])

    for sec in secs:
        sec_id = sec['id']
        secs.update_sec(sec_id, opt_mkt_prob=p.compute_heston_prob(sec_id))
        print(b.try_bet(sec_id))

if __name__ == '__main__':
    main()
