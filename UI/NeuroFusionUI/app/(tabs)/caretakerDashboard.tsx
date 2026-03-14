import CaretakerDashboardScreen from '../../src/screens/CaretakerDashboardScreen';
import { useNavigation } from '@react-navigation/native';

export default function CaretakerDashboard() {
  const navigation = useNavigation();
  return <CaretakerDashboardScreen navigation={navigation} />;
}