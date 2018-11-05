import pandas as pd

soi = ['448963772:3425425153:4459313622','448963763:4459313622:10437895','448963784:10437895:266482296','103524893:266482296:1195317079','502418741:1195317079:4929002368','502418742:4929002368:1242123670','103524902:1242123670:3396253208','103524902:3396253208:1242123651','103524902:1242123651:1195317083','448963776:1195317083:1242123649','448963789:1242123649:4459313610','448963780:4459313610:1280331077','448963780:1280331077:2599647834','448963787:2599647834:4459313608','448963760:4459313608:4459313607','448963783:4459313607:1065827673','448963765:1065827673:4459313603','87398581:4459313603:8121556','87398581:8121556:8121568','87398581:8121568:8121572','87398581:8121572:1411071138','87398581:1411071138:8121576','87398581:8121576:8121711','87398581:8121711:8121557','87398581:8121557:1696110319','87398581:1696110319:21788483','87398581:21788483:1282580772','87398581:1282580772:18165894','87398581:18165894:4927557854','502197868:4927557854:1696110295','448963774:1696110295:17710382','448963778:17710382:1085760670','448963771:1085760670:1277414337','241071582:1277414337:1277414356','241071582:1277414356:24950457','241071582:24950457:104494208','241071582:104494208:21788474','241071582:21788474:104494208','241071582:104494208:24950457','241071582:24950457:1277414356','241071582:1277414356:1277414337','502197869:1277414337:4927557855','502714814:4927557855:4927557854','87398581:4927557854:18165894','87398581:18165894:1282580772','87398581:1282580772:21788483','87398581:21788483:1696110319','87398581:1696110319:8121557','87398581:8121557:8121711','87398581:8121711:8121576','87398581:8121576:1411071138','87398581:1411071138:8121572','87398581:8121572:8121568','87398581:8121568:8121556','87398581:8121556:4459313603','448963765:4459313603:1065827673','448963785:1065827673:4459313602','448963768:4459313602:1195317078','448963790:1195317078:295133659','500955362:295133659:73095632','500955362:73095632:73099256','87397711:73099256:18165903','87397711:18165903:20908384','87397711:20908384:8081514','448963786:8081514:73082170','26176484:73082170:4929002367','26176484:4929002367:266482296','448963784:266482296:10437895','448963763:10437895:4459313622','448963772:4459313622:3425425153','448963755:4459313608:4459313617','161632579:4459313617:295133661','161632579:295133661:576724','161632579:576724:8121560','161632579:8121560:576725','25912893:576725:282678754','115678957:282678754:1306239519','265654074:1306239519:1306239263','115678954:1306239263:576726','78412354:576726:18165915','78412354:18165915:292907347','8122758:292907347:792846','78412355:792846:319833065','78412355:319833065:20908177','237099598:20908177:1043107528','237099598:1043107528:323206292','162089321:323206292:1649562464','87341562:1649562464:1649562493','87341562:1649562493:10776943','87341562:10776943:20908184','162089324:20908184:1649462945','162089324:1649462945:1649384672','162089324:1649384672:1649384611','87451616:2025484761:1017928632','87451616:1017928632:1306239377','87451616:1306239377:1306239666','87451616:1306239666:4121323609','87451616:4121323609:4929002366','502418739:4929002366:4929002367','502418740:4929002367:4929002368','162089324:1649384611:1649384672','162089324:1649384672:1649462945','162089324:1649462945:20908184','87341562:20908184:10776943','87341562:10776943:1649562493','87341562:1649562493:1649562464','162089321:1649562464:323206292','237099598:323206292:1043107528','237099598:1043107528:20908177','78412355:20908177:319833065','78412355:319833065:792846','8122758:792846:292907347','78412354:292907347:18165915','78412354:18165915:576726','115678974:576726:10939966','115678974:10939966:282678730','25912891:282678730:289783814','265654073:289783814:282678754','25912893:282678754:576725','161632579:576725:8121560','161632579:8121560:576724','161632579:576724:295133661','161632579:295133661:4459313617','448963758:4459313617:4459313607','448963753:4459313607:4459313602','502418738:73082170:4929002366','87451616:4929002366:4121323609','87451616:4121323609:1306239666','87451616:1306239666:1306239377','87451616:1306239377:1017928632','87451616:1017928632:2025484761','40667245:792846:4393824872','40667245:4393824872:792845','40667245:792845:20908385','40667245:20908385:20908178','40667245:20908178:792844','87724803:792844:20908158','87724803:20908158:20908159','39899568:20908159:20908160','39899568:20908160:20908161','87341541:20908161:20908162','87341541:20908162:792840','87341541:792840:20908162','87341541:20908162:20908161','39899568:20908161:20908160','39899568:20908160:20908159','87724803:20908159:20908158','87724803:20908158:792844','40667245:792844:20908178','40667245:20908178:20908385','40667245:20908385:792845','40667245:792845:4393824872','40667245:4393824872:792846','108233940:4459313602:4279694306','428851688:4279694306:4279694304','428851687:4279694304:576722','428851674:576722:4279694300','428851677:4279694300:4279694299','318929337:4279694299:20908383','428851681:20908383:4279694298','428851683:4279694298:4279694296','428851678:4279694296:279992577','428851680:279992577:4279694295','87502481:4279694295:4279694293','428851673:4279694293:287560137','428851675:287560137:4279694292','428851682:4279694292:1614362080','87502482:1614362080:1614362060','428851684:1614362060:576713','35085344:576713:3675352117','35205527:3675352117:576718','428851686:576718:576717','428851686:576717:576718','35205527:576718:3675352117','35085344:3675352117:576713','428851684:576713:1614362060','87502482:1614362060:1614362080','428851682:1614362080:4279694292','428851675:4279694292:287560137','428851673:287560137:4279694293','87502481:4279694293:4279694295','428851680:4279694295:279992577','428851678:279992577:4279694296','428851683:4279694296:4279694298','428851681:4279694298:20908383','318929337:20908383:4279694299','428851677:4279694299:4279694300','428851674:4279694300:576722','108233932:576722:4279694303','428851676:4279694303:2599647200','428851685:2599647200:4279694305','428851679:4279694305:4279694307','108233935:4279694307:1195317078','448963754:1195317078:4459313608']

def parse_data(files):
        
    df = pd.concat((pd.read_csv(f) for f in files))
    df = df[df['LinkRef'].isin(soi)]
    
    data_filter = df['JourneyRef'].str.extract('(?P<OperatingDayDate>\d{8})L(?P<LineNumber>\d{4})')['LineNumber'].astype(float).isin([1, 4, 8])
    data_filter &= df['StopPointRef'].isnull()
    df = df[data_filter].copy()
    
    df['wayId'], df['source'], df['target'] = df['LinkRef'].str.split(':').str

    #Convert time column to datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    #add an hour to convert to UTC+1
    df['Time'] += pd.to_timedelta(2,unit='h')
    #Set timestamp to index, allowing for interval aggregation
    df = df.set_index('Time')
    #Remove data during the night for now
    df = df.between_time('06:00','22:00')
    
    #Z-normalize data
    df['Speed'] = (df['Speed'] - df['Speed'].mean())/df['Speed'].std()
    
    
    df_5min = df.groupby([pd.Grouper(freq='5Min'),'wayId'])['Speed'].mean().reset_index(name='mean')

    #Make column with minutes and hours
    df_5min['hour'] = df_5min.Time.apply(lambda x: x.hour)
    df_5min['minute'] = df_5min.Time.apply(lambda x: x.minute)
        
    #Find the max number of minutes
    total_time = np.max(df_5min.hour*60 + df_5min.minute)
    
    #Add column with normalized cumulative minutes
    df_5min['NormalizedCumulativeMinutes'] = (df_5min.hour*60 + df_5min.minute) / total_time

    return df_5min

def vectorized_data(df):
    return df.pivot(index='Time', columns='wayId', values='mean')

def load_network():
    road_network = gpd.read_file('../data/road_network.geojson').drop('id', axis = 1)
    forward = road_network.copy().drop(['Oneway', 'MaxSpeedBackward'], axis = 1).rename({'MaxSpeedForward': 'MaxSpeed'}, axis = 1)
    forward['Heading'] = 'Forward'
    backward = road_network[lambda x: x['Oneway'] == 0].copy().drop(['Oneway', 'MaxSpeedForward'], axis = 1).rename({'Source': 'Target', 'Target': 'Source', 'MaxSpeedBackward': 'MaxSpeed'}, axis = 1)
    backward['geometry'] = backward['geometry'].apply(lambda x: LineString(reversed(x.coords)))
    backward['Heading'] = 'Backward'
    road_network = forward.append(
        backward,
        sort = True
    )
    road_network['LinkRef'] = road_network.apply(lambda r: '{WayId}:{Source}:{Target}'.format(**r), axis = 1)
    road_network.set_index('LinkRef', inplace = True)
