import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import numpy as np

class MoviaBusDataset(Dataset):
    """Dataset for Movia Bus data"""

    #Sections of interested in the road network
    section_of_interest = ['448963772:3425425153:4459313622','448963763:4459313622:10437895','448963784:10437895:266482296','103524893:266482296:1195317079','502418741:1195317079:4929002368','502418742:4929002368:1242123670','103524902:1242123670:3396253208','103524902:3396253208:1242123651','103524902:1242123651:1195317083','448963776:1195317083:1242123649','448963789:1242123649:4459313610','448963780:4459313610:1280331077','448963780:1280331077:2599647834','448963787:2599647834:4459313608','448963760:4459313608:4459313607','448963783:4459313607:1065827673','448963765:1065827673:4459313603','87398581:4459313603:8121556','87398581:8121556:8121568','87398581:8121568:8121572','87398581:8121572:1411071138','87398581:1411071138:8121576','87398581:8121576:8121711','87398581:8121711:8121557','87398581:8121557:1696110319','87398581:1696110319:21788483','87398581:21788483:1282580772','87398581:1282580772:18165894','87398581:18165894:4927557854','502197868:4927557854:1696110295','448963774:1696110295:17710382','448963778:17710382:1085760670','448963771:1085760670:1277414337','241071582:1277414337:1277414356','241071582:1277414356:24950457','241071582:24950457:104494208','241071582:104494208:21788474','241071582:21788474:104494208','241071582:104494208:24950457','241071582:24950457:1277414356','241071582:1277414356:1277414337','502197869:1277414337:4927557855','502714814:4927557855:4927557854','87398581:4927557854:18165894','87398581:18165894:1282580772','87398581:1282580772:21788483','87398581:21788483:1696110319','87398581:1696110319:8121557','87398581:8121557:8121711','87398581:8121711:8121576','87398581:8121576:1411071138','87398581:1411071138:8121572','87398581:8121572:8121568','87398581:8121568:8121556','87398581:8121556:4459313603','448963765:4459313603:1065827673','448963785:1065827673:4459313602','448963768:4459313602:1195317078','448963790:1195317078:295133659','500955362:295133659:73095632','500955362:73095632:73099256','87397711:73099256:18165903','87397711:18165903:20908384','87397711:20908384:8081514','448963786:8081514:73082170','26176484:73082170:4929002367','26176484:4929002367:266482296','448963784:266482296:10437895','448963763:10437895:4459313622','448963772:4459313622:3425425153','448963755:4459313608:4459313617','161632579:4459313617:295133661','161632579:295133661:576724','161632579:576724:8121560','161632579:8121560:576725','25912893:576725:282678754','115678957:282678754:1306239519','265654074:1306239519:1306239263','115678954:1306239263:576726','78412354:576726:18165915','78412354:18165915:292907347','8122758:292907347:792846','78412355:792846:319833065','78412355:319833065:20908177','237099598:20908177:1043107528','237099598:1043107528:323206292','162089321:323206292:1649562464','87341562:1649562464:1649562493','87341562:1649562493:10776943','87341562:10776943:20908184','162089324:20908184:1649462945','162089324:1649462945:1649384672','162089324:1649384672:1649384611','87451616:2025484761:1017928632','87451616:1017928632:1306239377','87451616:1306239377:1306239666','87451616:1306239666:4121323609','87451616:4121323609:4929002366','502418739:4929002366:4929002367','502418740:4929002367:4929002368','162089324:1649384611:1649384672','162089324:1649384672:1649462945','162089324:1649462945:20908184','87341562:20908184:10776943','87341562:10776943:1649562493','87341562:1649562493:1649562464','162089321:1649562464:323206292','237099598:323206292:1043107528','237099598:1043107528:20908177','78412355:20908177:319833065','78412355:319833065:792846','8122758:792846:292907347','78412354:292907347:18165915','78412354:18165915:576726','115678974:576726:10939966','115678974:10939966:282678730','25912891:282678730:289783814','265654073:289783814:282678754','25912893:282678754:576725','161632579:576725:8121560','161632579:8121560:576724','161632579:576724:295133661','161632579:295133661:4459313617','448963758:4459313617:4459313607','448963753:4459313607:4459313602','502418738:73082170:4929002366','87451616:4929002366:4121323609','87451616:4121323609:1306239666','87451616:1306239666:1306239377','87451616:1306239377:1017928632','87451616:1017928632:2025484761','40667245:792846:4393824872','40667245:4393824872:792845','40667245:792845:20908385','40667245:20908385:20908178','40667245:20908178:792844','87724803:792844:20908158','87724803:20908158:20908159','39899568:20908159:20908160','39899568:20908160:20908161','87341541:20908161:20908162','87341541:20908162:792840','87341541:792840:20908162','87341541:20908162:20908161','39899568:20908161:20908160','39899568:20908160:20908159','87724803:20908159:20908158','87724803:20908158:792844','40667245:792844:20908178','40667245:20908178:20908385','40667245:20908385:792845','40667245:792845:4393824872','40667245:4393824872:792846','108233940:4459313602:4279694306','428851688:4279694306:4279694304','428851687:4279694304:576722','428851674:576722:4279694300','428851677:4279694300:4279694299','318929337:4279694299:20908383','428851681:20908383:4279694298','428851683:4279694298:4279694296','428851678:4279694296:279992577','428851680:279992577:4279694295','87502481:4279694295:4279694293','428851673:4279694293:287560137','428851675:287560137:4279694292','428851682:4279694292:1614362080','87502482:1614362080:1614362060','428851684:1614362060:576713','35085344:576713:3675352117','35205527:3675352117:576718','428851686:576718:576717','428851686:576717:576718','35205527:576718:3675352117','35085344:3675352117:576713','428851684:576713:1614362060','87502482:1614362060:1614362080','428851682:1614362080:4279694292','428851675:4279694292:287560137','428851673:287560137:4279694293','87502481:4279694293:4279694295','428851680:4279694295:279992577','428851678:279992577:4279694296','428851683:4279694296:4279694298','428851681:4279694298:20908383','318929337:20908383:4279694299','428851677:4279694299:4279694300','428851674:4279694300:576722','108233932:576722:4279694303','428851676:4279694303:2599647200','428851685:2599647200:4279694305','428851679:4279694305:4279694307','108233935:4279694307:1195317078','448963754:1195317078:4459313608']
    num_roads = len(section_of_interest)
    def __init__(self, \
                root_dir, \
                prev_timesteps = 0, \
                verbose = False, \
                interpolation = False, \
                normalize = False, \
                max_future_time_steps = 1, \
                timeofday = False, \
                sequence_target = False
                ):
        """
        
        Args:
        ---
            root_dir (string): Root directory of the data. Module will automatically find all bus data in subfolders of this directory
            prev_timesteps (Int): Number of prior timesteps to give for each data point
            verbose (Boolean): Flag if you want to print the progress of loading the files
            interpolation (Boolean): Flag if you want to interpolate missing datapoints
            normalize (Boolean): Flag if you want the data to be normalized
            max_future_time_steps (Int): The maximum number of time steps into the future we want to predict.
            timeofday (Boolean): Flag if you want the data to include the time of day as a continous value between 0 and 1
            sequence_target (Boolean): Flag if you want the target data to be given as a sequence of all timesteps up to max_future_time_steps
        """
        
        self.dataframes = []
        self.mean = None
        self.std = None
        self.__data_per_dataframe = 0
        self.__prev_timesteps = prev_timesteps
        self.__interpolation = interpolation
        self.__max_future_time_steps = max_future_time_steps
        self.__timeofday = timeofday
        self.__sequence_target = sequence_target

        #find all bus data in the given directory
        files = glob.glob('{}/*/vehicle-position-matched-online.csv'.format(root_dir))

        for file in files:
            if verbose:
                print('Loading file {}...'.format(file))
            dataframe = self.__parse_file(file)

            #Each dataset contrains n points, but we need some for the previous steps, some for the future time steps, and  1 to conver from length to index
            self.__data_per_dataframe = len(dataframe) - self.__prev_timesteps - self.__max_future_time_steps 
            self.dataframes.append(dataframe)
        
        if normalize:
            self.normalize()


    def __parse_file(self, file):
        df = pd.read_csv(file)
        #filter out sections of interest
        df = df[df['LinkRef'].isin(self.section_of_interest)]

        #Only consider 1A, 4A and 8A busses
        data_filter = df['JourneyRef'].str.extract('(?P<OperatingDayDate>\d{8})L(?P<LineNumber>\d{4})')['LineNumber'].astype(float).isin([1, 4, 8])
        
        #Remove busses at stopping points
        data_filter &= df['StopPointRef'].isnull()
        df = df[data_filter].copy()
        
        #Convert time column to datetime
        df['Time'] = pd.to_datetime(df['Time'])
        
        #add an hour to convert to UTC+1
        df['Time'] += pd.to_timedelta(2,unit='h')

        #Set timestamp to index, allowing for interval aggregation
        df = df.set_index('Time')

        #Remove data during the night for now
        df = df.between_time('06:00','22:00')
        
        #Aggregate data for each road into 5min bins.
        df_5min = df.groupby([pd.Grouper(freq='5Min'),'LinkRef'])['Speed'].mean().reset_index(name='Speed')

        df_5min = df_5min.pivot(index='Time', columns='LinkRef', values='Speed')

        if self.__timeofday:
            #Add a column indicating time of day between 0 and 1
            df_5min['TimeOfDay'] = (df_5min.index.minute + df_5min.index.hour*60)/(22*60)

        #Interpolate missing values
        if self.__interpolation:
            df_5min.interpolate(limit=15, limit_direction='both',inplace=True)   

        return df_5min

    def normalize(self, mean=None, std=None, individual_roads=True):
        """
        Normalizes the data set. If no arguments is given, the empirical mean and standard deviation from the dataset is used.
        Otherwise the provided mean and standard deviation is used

        Args:
            mean (float): The mean to be used for normalization. 
            std (float): The standard deviation to be used for normalization
        """
        #If mean and std is not given, calculate the emperical values
        if mean is None and std is None:
            if self.__timeofday:
                df = pd.concat((dataframe.drop('TimeOfDay', axis=1) for dataframe in self.dataframes))
            else:
                df = pd.concat((dataframe for dataframe in self.dataframes))

            if individual_roads:
                self.mean = df.mean()
                self.std = df.std()
            else:
                self.mean = np.nanmean(df.values)
                self.std = np.nanstd(df.values)
        #Otherwise set the given values for mean and std
        elif mean is not None and std is not None:
            self.mean = mean
            self.std = std

        #Z-normalize the dataframes one by one
        for i in range(len(self.dataframes)):
            self.dataframes[i] = (self.dataframes[i]-self.mean)/self.std

            if self.__timeofday:
                #Instead of trying to exclude the time of day from normalization, just recalculate it, since this is way easier...
                self.dataframes[i]['TimeOfDay'] = (self.dataframes[i].index.minute + self.dataframes[i].index.hour*60)/(22*60)


    def __len__(self):
        #Each dataframe has data from 06:00 - 22:00, in steps of 5, so we have 193 data points.
        return len(self.dataframes) * self.__data_per_dataframe

    def __getitem__(self, idx):
        """Used for pytorch DataSet, to find the entry for a given index"""
        #Calculate which dataframe to use
        dataframe_idx = idx // self.__data_per_dataframe

        #Calculate the index within the dataframe. The +1 is due to the fact that we want to index a slice, where the last element is disregarded
        idx = (idx % self.__data_per_dataframe) + self.__prev_timesteps + 1
        #Add the additional prediction steps, and remove the 1 since we now want a real index and not a slice
        idx_target = idx + self.__max_future_time_steps - 1

        #Get the data for idx and previous time steps
        data = torch.tensor(self.dataframes[dataframe_idx] [idx - self.__prev_timesteps - 1 : idx ].values, dtype=torch.float)
        
        
        if self.__sequence_target:
            target =torch.tensor(self.dataframes[dataframe_idx] [idx : idx_target + 1].values, dtype=torch.float)[:,0:self.num_roads]    
        else:
        
            target =torch.tensor(self.dataframes[dataframe_idx].iloc[idx_target].values, dtype=torch.float)[0:self.num_roads]
        
        return {'data':data, 'target':target}

if __name__ == "__main__":
    train = MoviaBusDataset('data/train', interpolation=True, \
                        prev_timesteps=3, \
                        max_future_time_steps=5, \
                        sequence_target=True)
    train[0]