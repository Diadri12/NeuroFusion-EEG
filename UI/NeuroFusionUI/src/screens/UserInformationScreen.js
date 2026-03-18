import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, StatusBar,
  SafeAreaView, ScrollView, Animated,
  TouchableOpacity, TextInput, Alert, ActivityIndicator,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { onAuthStateChanged, updateProfile } from 'firebase/auth';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { auth, db } from '../config/firebase';
 
const UserInformationScreen = () => {
  const router = useRouter();
 
  const [user,        setUser]        = useState(null);
  const [displayName, setDisplayName] = useState('');
  const [email,       setEmail]       = useState('');
  const [role,        setRole]        = useState('patient');
  const [joined,      setJoined]      = useState('');
  const [editing,     setEditing]     = useState(false);
  const [newName,     setNewName]     = useState('');
  const [saving,      setSaving]      = useState(false);
  const [loading,     setLoading]     = useState(true);
 
  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;
 
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        setEmail(currentUser.email || '');
        setDisplayName(currentUser.displayName || currentUser.email?.split('@')[0] || '');
        setNewName(currentUser.displayName || '');
 
        // Format join date
        const created = currentUser.metadata?.creationTime;
        if (created) {
          setJoined(new Date(created).toLocaleDateString('en-US', {
            year: 'numeric', month: 'long', day: 'numeric',
          }));
        }
 
        // Get role from Firestore
        try {
          const snap = await getDoc(doc(db, 'users', currentUser.uid));
          if (snap.exists()) {
            const fetchedRole = snap.data()?.role;
            if (fetchedRole) setRole(fetchedRole);
          }
        } catch (e) {
          console.log('Role fetch error:', e);
        }
      }
      setLoading(false);
    });
 
    Animated.parallel([
      Animated.timing(fadeAnim,  { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.spring(slideAnim, { toValue: 0, friction: 8,   useNativeDriver: true }),
    ]).start();
 
    return unsubscribe;
  }, []);
 
  const roleColour = () => {
    if (role === 'doctor')    return { bg: '#EEF2FF', text: '#4F46E5' };
    if (role === 'caretaker') return { bg: '#FFF7ED', text: '#EA580C' };
    return                           { bg: '#F0FFF4', text: '#16A34A' };
  };
 
  const roleIcon = () => {
    if (role === 'doctor')    return 'doctor';
    if (role === 'caretaker') return 'account-heart';
    return 'account';
  };
 
  const saveName = async () => {
    if (!newName.trim()) {
      Alert.alert('Error', 'Display name cannot be empty.');
      return;
    }
    setSaving(true);
    try {
      await updateProfile(user, { displayName: newName.trim() });
      await updateDoc(doc(db, 'users', user.uid), { displayName: newName.trim() });
      setDisplayName(newName.trim());
      setEditing(false);
      Alert.alert('Saved', 'Your display name has been updated.');
    } catch (e) {
      Alert.alert('Error', 'Could not update display name. Please try again.');
    } finally {
      setSaving(false);
    }
  };
 
  if (loading) {
    return (
      <View style={styles.loadingWrap}>
        <ActivityIndicator size="large" color="#B844FF" />
      </View>
    );
  }
 
  const { bg, text } = roleColour();
 
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#B844FF" />
 
      {/* Header */}
      <Animated.View style={[styles.header, { opacity: fadeAnim }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#FFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>User Information</Text>
        <View style={{ width: 36 }} />
      </Animated.View>
 
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
 
          {/* Avatar + name */}
          <View style={styles.heroCard}>
            <View style={styles.avatarWrap}>
              <MaterialCommunityIcons name={roleIcon()} size={48} color="#B844FF" />
            </View>
            <Text style={styles.heroName}>{displayName}</Text>
            <Text style={styles.heroEmail}>{email}</Text>
            <View style={[styles.roleBadge, { backgroundColor: bg }]}>
              <Text style={[styles.roleBadgeText, { color: text }]}>
                {role.charAt(0).toUpperCase() + role.slice(1)}
              </Text>
            </View>
          </View>
 
          {/* Account details */}
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <MaterialCommunityIcons name="account-details" size={18} color="#B844FF" />
              <Text style={styles.cardTitle}>Account Details</Text>
            </View>
 
            {/* Display name row */}
            <View style={styles.fieldRow}>
              <MaterialCommunityIcons name="pencil-circle" size={20} color="#B844FF" />
              <View style={styles.fieldBody}>
                <Text style={styles.fieldLabel}>Display Name</Text>
                {editing ? (
                  <TextInput
                    style={styles.fieldInput}
                    value={newName}
                    onChangeText={setNewName}
                    autoFocus
                    placeholder="Enter display name"
                    placeholderTextColor="#CCC"
                  />
                ) : (
                  <Text style={styles.fieldValue}>{displayName}</Text>
                )}
              </View>
              {editing ? (
                <View style={styles.editActions}>
                  <TouchableOpacity onPress={() => { setEditing(false); setNewName(displayName); }}>
                    <MaterialCommunityIcons name="close" size={20} color="#999" />
                  </TouchableOpacity>
                  <TouchableOpacity onPress={saveName} disabled={saving}>
                    {saving
                      ? <ActivityIndicator size="small" color="#B844FF" />
                      : <MaterialCommunityIcons name="check" size={20} color="#B844FF" />
                    }
                  </TouchableOpacity>
                </View>
              ) : (
                <TouchableOpacity onPress={() => setEditing(true)}>
                  <MaterialCommunityIcons name="pencil" size={18} color="#B844FF" />
                </TouchableOpacity>
              )}
            </View>
 
            <View style={[styles.fieldRow, { borderBottomWidth: 0 }]}>
              <MaterialCommunityIcons name="email-outline" size={20} color="#888" />
              <View style={styles.fieldBody}>
                <Text style={styles.fieldLabel}>Email</Text>
                <Text style={styles.fieldValue}>{email}</Text>
              </View>
            </View>
          </View>
 
          {/* Role & membership */}
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <MaterialCommunityIcons name="shield-account" size={18} color="#B844FF" />
              <Text style={styles.cardTitle}>Role & Membership</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Role</Text>
              <Text style={[styles.infoValue, { color: text }]}>
                {role.charAt(0).toUpperCase() + role.slice(1)}
              </Text>
            </View>
            <View style={[styles.infoRow, { borderBottomWidth: 0 }]}>
              <Text style={styles.infoLabel}>Member since</Text>
              <Text style={styles.infoValue}>{joined || '—'}</Text>
            </View>
          </View>
 
          {/* Security */}
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <MaterialCommunityIcons name="lock" size={18} color="#B844FF" />
              <Text style={styles.cardTitle}>Security</Text>
            </View>
            <View style={[styles.infoRow, { borderBottomWidth: 0 }]}>
              <Text style={styles.infoLabel}>Password</Text>
              <Text style={styles.infoValue}>••••••••</Text>
            </View>
          </View>
 
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
};
 
const styles = StyleSheet.create({
  container:    { flex: 1, backgroundColor: '#F4F5F9' },
  loadingWrap:  { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FFF' },
 
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: '#B844FF', paddingHorizontal: 18, paddingVertical: 14,
  },
  backBtn: {
    width: 36, height: 36, borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center', alignItems: 'center',
  },
  headerTitle: { fontSize: 18, fontWeight: '800', color: '#FFF' },
 
  scroll: { padding: 18, paddingBottom: 60 },
 
  heroCard: {
    backgroundColor: '#FFF', borderRadius: 20, padding: 28,
    alignItems: 'center', marginBottom: 14,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07, shadowRadius: 8, elevation: 3,
  },
  avatarWrap: {
    width: 88, height: 88, borderRadius: 44,
    backgroundColor: '#F5EEFF',
    justifyContent: 'center', alignItems: 'center', marginBottom: 12,
  },
  heroName:  { fontSize: 22, fontWeight: '800', color: '#1A1A2E', marginBottom: 4 },
  heroEmail: { fontSize: 13, color: '#888', marginBottom: 12 },
  roleBadge: { borderRadius: 20, paddingHorizontal: 16, paddingVertical: 5 },
  roleBadgeText: { fontSize: 12, fontWeight: '700' },
 
  card: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 18, marginBottom: 14,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06, shadowRadius: 6, elevation: 2,
  },
  cardTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 14 },
  cardTitle:    { fontSize: 15, fontWeight: '700', color: '#1A1A2E' },
 
  fieldRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#F5F5F5',
  },
  fieldBody:   { flex: 1 },
  fieldLabel:  { fontSize: 11, color: '#AAA', marginBottom: 2 },
  fieldValue:  { fontSize: 14, fontWeight: '600', color: '#333' },
  fieldInput:  { fontSize: 14, fontWeight: '600', color: '#333', borderBottomWidth: 1, borderBottomColor: '#B844FF', paddingVertical: 2 },
  editActions: { flexDirection: 'row', gap: 10, alignItems: 'center' },
 
  infoRow: {
    flexDirection: 'row', justifyContent: 'space-between',
    paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#F5F5F5',
  },
  infoLabel: { fontSize: 13, color: '#888' },
  infoValue: { fontSize: 13, fontWeight: '700', color: '#333' },
});
 
export default UserInformationScreen;