import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

// Configure how notifications should be displayed when app is in foreground
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

class NotificationService {
  constructor() {
    this.notificationListener = null;
    this.responseListener = null;
  }

  // Request notification permissions
  async requestPermissions() {
    try {
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;

      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }

      if (finalStatus !== 'granted') {
        console.log('Failed to get push token for push notification!');
        return false;
      }

      return true;
    } catch (error) {
      console.error('Error requesting notification permissions:', error);
      return false;
    }
  }

  // Schedule immediate notification for seizure detection
  async sendSeizureAlert() {
    try {
      const hasPermission = await this.requestPermissions();
      
      if (!hasPermission) {
        console.log('Notification permission not granted');
        return null;
      }

      const notificationId = await Notifications.scheduleNotificationAsync({
        content: {
          title: '⚠️ Seizure Detected',
          body: 'A Seizure is detected, please take necessary precautions',
          sound: true,
          priority: Notifications.AndroidNotificationPriority.HIGH,
          color: '#E63946',
          badge: 1,
          data: { 
            type: 'seizure_alert',
            timestamp: new Date().toISOString(),
          },
        },
        trigger: null, // Show immediately
      });

      console.log('Seizure notification sent:', notificationId);
      return notificationId;
    } catch (error) {
      console.error('Error sending seizure notification:', error);
      return null;
    }
  }

  // Schedule notification for no seizure (optional, for testing)
  async sendNoSeizureNotification() {
    try {
      const hasPermission = await this.requestPermissions();
      
      if (!hasPermission) {
        return null;
      }

      const notificationId = await Notifications.scheduleNotificationAsync({
        content: {
          title: 'No Seizure Detected',
          body: 'Analysis complete. No seizure activity detected.',
          sound: true,
          priority: Notifications.AndroidNotificationPriority.DEFAULT,
          color: '#00D66A',
          badge: 1,
          data: { 
            type: 'no_seizure',
            timestamp: new Date().toISOString(),
          },
        },
        trigger: null,
      });

      return notificationId;
    } catch (error) {
      console.error('Error sending no seizure notification:', error);
      return null;
    }
  }

  // Set up notification listeners
  setupNotificationListeners(onNotificationReceived, onNotificationResponse) {
    // Listener for notifications received while app is in foreground
    this.notificationListener = Notifications.addNotificationReceivedListener(
      (notification) => {
        console.log('Notification received:', notification);
        if (onNotificationReceived) {
          onNotificationReceived(notification);
        }
      }
    );

    // Listener for when user interacts with notification
    this.responseListener = Notifications.addNotificationResponseReceivedListener(
      (response) => {
        console.log('Notification response:', response);
        if (onNotificationResponse) {
          onNotificationResponse(response);
        }
      }
    );
  }

  // Remove notification listeners
  removeNotificationListeners() {
    if (this.notificationListener) {
      Notifications.removeNotificationSubscription(this.notificationListener);
    }
    if (this.responseListener) {
      Notifications.removeNotificationSubscription(this.responseListener);
    }
  }

  // Cancel all notifications
  async cancelAllNotifications() {
    await Notifications.cancelAllScheduledNotificationsAsync();
  }

  // Get badge count
  async getBadgeCount() {
    return await Notifications.getBadgeCountAsync();
  }

  // Set badge count
  async setBadgeCount(count) {
    await Notifications.setBadgeCountAsync(count);
  }

  // Clear badge
  async clearBadge() {
    await Notifications.setBadgeCountAsync(0);
  }
}

export default new NotificationService();

