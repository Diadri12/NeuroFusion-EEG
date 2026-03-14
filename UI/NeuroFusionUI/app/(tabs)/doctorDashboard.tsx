import { useNavigation } from '@react-navigation/native';
import DoctorDashboardScreen from '../../src/screens/DoctorDashboardScreen';

export default function DoctorDashboard() {
  const navigation = useNavigation();
  return <DoctorDashboardScreen navigation={navigation} />;
}