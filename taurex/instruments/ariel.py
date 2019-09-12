from taurex.instruments.instrument import Instrument


class ArielInstrument(Instrument):


    def __init__(self, config_file = None):
        super().__init__()

        try:
            from ArielRad.target_list import TargetList
        except ImportError:
            self.error('')

        if config_file is None:
            self.error('No config file given for ariel rad')
            raise ValueError('No config file given')
            
        